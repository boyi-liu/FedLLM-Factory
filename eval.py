"""
Standalone post-training evaluation script.

Loads the global LoRA adapter saved after federated training and evaluates
on each client's test split using the full metric suite configured in
utils/eval.yaml.

Usage:
    python eval.py --suffix default --alg fedit --model <model_name_or_path> \
                   --dataset gsm8k --cn 10 --epoch 5 --lr 1e-4

Which metrics are computed is controlled entirely by utils/eval.yaml (metrics field
per dataset).  No extra flags needed — add/remove entries in the yaml to opt in/out.
"""

import importlib
import math
import os

import torch
from datasets import load_dataset
from peft import get_peft_model
from rouge_score import rouge_scorer
from torch.amp import autocast
from torch.utils.data import DataLoader

from utils.eval_utils import get_dataset_config, _normalize, _load_eval_config
from utils.model_utils import load_model, load_tokenizer, load_lora_config
from utils.data_utils import load_data
from utils.options import build_parser
from utils.logger import get_logger


_EVAL_CONFIG = _load_eval_config()


# ------------------------------------------------------------------
# Adapter path
# ------------------------------------------------------------------

def _adapter_path(args) -> str:
    base = f'{args.suffix}'
    name = (
        f'{args.alg}_{args.dataset}_{args.model}_'
        f'{args.cn}c_{args.epoch}E_lr{args.lr}'
    )
    return os.path.join(base, 'adapter', name)


# ------------------------------------------------------------------
# Shared generation helper
# ------------------------------------------------------------------

def _generate_predictions(model, tokenizer, args, client_idx: int) -> list:
    """Generate (pred_text, gold_label) pairs for one client's test split.

    Loads the raw JSONL (unpacked text) so the prompt contains only the
    question, not the answer.  Returns a flat list of (pred, gold) tuples.
    """
    raw_path = os.path.join('dataset', args.dataset, f'test/{client_idx}.jsonl')
    raw_ds = load_dataset('json', data_files={'test': raw_path})['test']
    final_samples = _EVAL_CONFIG.get('final_eval_samples', 0)
    if final_samples > 0:
        raw_ds = raw_ds.select(range(min(final_samples, len(raw_ds))))

    batch_size = _EVAL_CONFIG.get('final_eval_batch_size', 8)
    predictions = []
    for batch in DataLoader(raw_ds, batch_size=batch_size, shuffle=False):
        prompts = [f"Instruct: {inp}\nAnswer:" for inp in batch['input_ids']]
        enc = tokenizer(
            prompts, return_tensors='pt', truncation=True,
            padding=True, max_length=512,
        ).to(model.device)
        with torch.no_grad():
            pred_ids = model.generate(**enc, max_new_tokens=64)
        # decode only the newly generated tokens
        prompt_len = enc['input_ids'].shape[1]
        pred_texts = tokenizer.batch_decode(pred_ids[:, prompt_len:], skip_special_tokens=True)
        predictions.extend(zip(pred_texts, batch['label']))

    return predictions


# ------------------------------------------------------------------
# Generation-based metric evaluators
# ------------------------------------------------------------------

class _ExactMatchEvaluator:
    """Computes exact match from a list of (pred, gold) pairs."""

    def evaluate(self, predictions: list) -> dict:
        if not predictions:
            return {'exact_match': 0.0}
        matches = sum(int(_normalize(p) == _normalize(g)) for p, g in predictions)
        return {'exact_match': matches / len(predictions)}


class _RougeEvaluator:
    """Computes ROUGE-1/2/L from a list of (pred, gold) pairs."""

    def __init__(self):
        self._scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
        )

    def evaluate(self, predictions: list) -> dict:
        totals = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        if not predictions:
            return totals
        for pred, gold in predictions:
            scores = self._scorer.score(gold, pred)
            for k in totals:
                totals[k] += scores[k].fmeasure
        n = len(predictions)
        return {k: v / n for k, v in totals.items()}


class _F1Evaluator:
    """Token-level F1 from a list of (pred, gold) pairs (SQuAD-style).

    Tokens are obtained by normalising then splitting on whitespace.
    F1 = 2 * precision * recall / (precision + recall), where
      precision = |common| / |pred_tokens|
      recall    = |common| / |gold_tokens|
    """

    @staticmethod
    def _token_f1(pred: str, gold: str) -> float:
        pred_tokens = _normalize(pred).split()
        gold_tokens = _normalize(gold).split()
        if not pred_tokens or not gold_tokens:
            return float(pred_tokens == gold_tokens)
        common = len(set(pred_tokens) & set(gold_tokens))
        if common == 0:
            return 0.0
        precision = common / len(pred_tokens)
        recall = common / len(gold_tokens)
        return 2 * precision * recall / (precision + recall)

    def evaluate(self, predictions: list) -> dict:
        if not predictions:
            return {'f1': 0.0}
        total = sum(self._token_f1(p, g) for p, g in predictions)
        return {'f1': total / len(predictions)}


# ------------------------------------------------------------------
# MMLU benchmark evaluator
# ------------------------------------------------------------------

class _MMLUEvaluator:
    """
    Evaluates a model on the MMLU benchmark (cais/mmlu).
    Scoring: log-prob of each answer token (A/B/C/D) at the last position.
    Run once on the global model, not per client.
    """

    CHOICES = ['A', 'B', 'C', 'D']
    SUBJECT_CATEGORY = 'all'

    def __init__(self, tokenizer, n_shot: int = 5, max_test_per_subject: int = 100):
        self.n_shot = n_shot
        self.max_test = max_test_per_subject
        self.tokenizer = tokenizer
        print(f'[MMLU] Loading cais/mmlu ({self.SUBJECT_CATEGORY}) …')
        ds = load_dataset('cais/mmlu', self.SUBJECT_CATEGORY)
        self._dev = ds['validation']
        self._test = ds['test']
        self._choice_ids = [
            self.tokenizer.encode(f' {c}', add_special_tokens=False)[0]
            for c in self.CHOICES
        ]

    def evaluate(self, model) -> dict:
        model.eval()
        device = model.device
        subjects = list(set(self._test['subject']))
        subject_acc = {}
        for subject in subjects:
            dev_rows = [r for r in self._dev if r['subject'] == subject]
            few_shot_prefix = self._build_few_shot_prefix(dev_rows[:self.n_shot], subject)
            test_rows = [r for r in self._test if r['subject'] == subject][:self.max_test]
            correct = 0
            for row in test_rows:
                prompt = few_shot_prefix + self._format_question(row, include_answer=False)
                encoding = self.tokenizer(
                    prompt, return_tensors='pt', truncation=True,
                    max_length=2048, padding=False,
                ).to(device)
                with torch.no_grad(), autocast('cuda'):
                    logits = model(**encoding).logits
                choice_logits = logits[0, -1, self._choice_ids]
                pred = choice_logits.argmax().item()
                correct += int(pred == row['answer'])
            subject_acc[subject] = correct / len(test_rows) if test_rows else 0.0

        avg_acc = sum(subject_acc.values()) / len(subject_acc) if subject_acc else 0.0
        return {'mmlu_accuracy': avg_acc, **{f'mmlu_{s}': v for s, v in subject_acc.items()}}

    def _build_few_shot_prefix(self, dev_rows: list, subject: str) -> str:
        header = (
            f'The following are multiple choice questions (with answers) '
            f'about {subject.replace("_", " ")}.\n\n'
        )
        return header + ''.join(self._format_question(r, include_answer=True) for r in dev_rows)

    def _format_question(self, row: dict, include_answer: bool) -> str:
        choices_str = '\n'.join(f'{l}. {c}' for l, c in zip(self.CHOICES, row['choices']))
        q = f"Question: {row['question']}\n{choices_str}\nAnswer:"
        if include_answer:
            q += f" {self.CHOICES[row['answer']]}\n\n"
        return q


# ------------------------------------------------------------------
# Per-client evaluation
# ------------------------------------------------------------------

def _eval_client(model, tokenizer, client_idx: int, args, dataset_cfg: dict) -> dict:
    task_type = dataset_cfg['task_type']
    metrics = dataset_cfg['metrics']

    formatted_ds = load_data(args, idx=client_idx)['test']
    eval_samples = _EVAL_CONFIG.get('eval_samples', 0)
    if eval_samples > 0:
        formatted_ds = formatted_ds.select(range(min(eval_samples, len(formatted_ds))))
    result = {}

    if task_type == 'SEQ_CLS':
        loader = DataLoader(formatted_ds, batch_size=1, shuffle=False)
        correct, total = 0, 0
        for batch in loader:
            input_ids = torch.stack(batch['input_ids']).transpose(0, 1).to(model.device)
            attention_mask = torch.stack(batch['attention_mask']).transpose(0, 1).to(model.device)
            labels = torch.tensor(batch['labels']).to(model.device)
            with torch.no_grad(), autocast('cuda'):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        result['accuracy'] = correct / total if total > 0 else 0.0

    elif task_type == 'CAUSAL_LM':
        # loss / perplexity
        loader = DataLoader(formatted_ds, batch_size=1, shuffle=False)
        total_loss, total_steps = 0.0, 0
        for batch in loader:
            input_ids = torch.stack(batch['input_ids']).transpose(0, 1).to(model.device)
            attention_mask = torch.stack(batch['attention_mask']).transpose(0, 1).to(model.device)
            labels = torch.stack(batch['labels']).transpose(0, 1).to(model.device)
            with torch.no_grad(), autocast('cuda'):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_loss += outputs.loss.item()
                total_steps += 1
        avg_loss = total_loss / total_steps if total_steps > 0 else 0.0
        result['eval_loss'] = avg_loss
        result['perplexity'] = math.exp(avg_loss) if avg_loss < 20 else float('inf')

        # generation-based metrics — one shared generation pass
        run_exact_match = 'exact_match' in metrics
        run_rouge = 'rouge' in metrics
        run_f1 = 'f1' in metrics
        if run_exact_match or run_rouge or run_f1:
            predictions = _generate_predictions(model, tokenizer, args, client_idx)
            if run_exact_match:
                result.update(_ExactMatchEvaluator().evaluate(predictions))
            if run_rouge:
                result.update(_RougeEvaluator().evaluate(predictions))
            if run_f1:
                result.update(_F1Evaluator().evaluate(predictions))

    return result


# ------------------------------------------------------------------
# Eval-specific arguments (follows the same add_args pattern as algorithms)
# ------------------------------------------------------------------

def add_args(parser):
    return parser.parse_args()


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    parser = build_parser()

    # First pass to get --alg, then load algorithm-specific args
    args, _ = parser.parse_known_args()
    alg_module = importlib.import_module(f'alg.{args.alg}')
    if hasattr(alg_module, 'add_args'):
        alg_module.add_args(parser)

    # Final parse
    args = add_args(parser)

    args.suffix = f'exp/{args.suffix}'
    logger = get_logger(args)

    # Derive task_type from dataset config
    dataset_cfg = get_dataset_config(args.dataset)
    args.task_type = dataset_cfg['task_type']

    adapter_path = _adapter_path(args)

    tokenizer = load_tokenizer(args)
    base_model = load_model(args)
    model = get_peft_model(base_model, load_lora_config(args))
    lora_weights = torch.load(
        os.path.join(adapter_path, 'lora_weights.pt'), map_location='cpu'
    )
    model.load_state_dict(lora_weights, strict=False)
    model.eval()

    # Per-client evaluation (all metrics except mmlu)
    all_metrics = []
    for cid in range(args.cn):
        metrics = _eval_client(model, tokenizer, cid, args, dataset_cfg)
        parts = ' | '.join(f'{k}: {v:.4f}' for k, v in metrics.items())
        print(f'[Client {cid}] {parts}')
        all_metrics.append(metrics)

    if all_metrics:
        agg = {k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in all_metrics[0]}
        parts = ' | '.join(f'{k}: {v:.4f}' for k, v in agg.items())
        print(f'[Avg] {parts}')
        logger.info(f'[Avg] {parts}')

    # MMLU — run once on the global model, triggered by eval.yaml config
    if 'mmlu' in dataset_cfg['metrics']:
        mmlu_cfg = _EVAL_CONFIG.get('global_benchmarks', {}).get('mmlu', {})
        mmlu_ev = _MMLUEvaluator(
            tokenizer,
            n_shot=mmlu_cfg.get('n_shot', 5),
            max_test_per_subject=mmlu_cfg.get('max_per_subject', 100),
        )
        mmlu_result = mmlu_ev.evaluate(model)
        logger.info(f"[MMLU] mmlu_accuracy: {mmlu_result['mmlu_accuracy']:.4f}")


if __name__ == '__main__':
    main()
