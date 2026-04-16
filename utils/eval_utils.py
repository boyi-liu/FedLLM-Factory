import math
import os
import re

import torch
import yaml
from torch.amp import autocast
from torch.utils.data import DataLoader


def _load_eval_config():
    config_path = os.path.join(os.path.dirname(__file__), 'eval.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


_EVAL_CONFIG = _load_eval_config()


def get_dataset_config(dataset_name: str) -> dict:
    """Return the eval config entry for a given dataset name."""
    datasets = _EVAL_CONFIG.get('datasets', {})
    if dataset_name not in datasets:
        raise ValueError(
            f"Dataset '{dataset_name}' not found in eval.yaml. "
            f"Available: {list(datasets.keys())}"
        )
    return datasets[dataset_name]


class Evaluator:
    """
    Stateless evaluator decoupled from Trainer.
    Instantiated once per client and reused across rounds.
    """

    def __init__(self, args, dataset):
        self.args = args
        self.dataset_cfg = get_dataset_config(args.dataset)
        self.task_type = self.dataset_cfg['task_type']
        self.metrics = self.dataset_cfg['metrics']
        self.primary_metric = self.dataset_cfg['primary_metric']

        eval_samples = _EVAL_CONFIG.get('eval_samples', 0)
        test_data = dataset['test']
        if eval_samples > 0:
            test_data = test_data.select(range(min(eval_samples, len(test_data))))

        self.eval_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate(self, model, round_idx=None, client_id=None) -> dict:
        """Run evaluation and return a metrics dict."""
        model.eval()
        if self.task_type == 'SEQ_CLS':
            result = self._eval_seq_cls(model)
        elif self.task_type == 'CAUSAL_LM':
            result = self._eval_causal_lm(model)
        else:
            raise ValueError(f"Unsupported task_type: {self.task_type}")

        self._log(result, round_idx, client_id)
        return result

    # ------------------------------------------------------------------
    # Per-task implementations
    # ------------------------------------------------------------------

    def _eval_seq_cls(self, model) -> dict:
        correct = 0
        total = 0
        for batch in self.eval_loader:
            input_ids = torch.stack(batch['input_ids']).transpose(0, 1).to(model.device)
            attention_mask = torch.stack(batch['attention_mask']).transpose(0, 1).to(model.device)
            labels = torch.tensor(batch['labels']).to(model.device)
            with torch.no_grad(), autocast('cuda'):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        accuracy = correct / total if total > 0 else 0.0
        return {'accuracy': accuracy}

    def _eval_causal_lm(self, model) -> dict:
        total_loss = 0.0
        total_steps = 0
        exact_matches = 0
        total_samples = 0
        run_exact_match = 'exact_match' in self.metrics

        for batch in self.eval_loader:
            input_ids = torch.stack(batch['input_ids']).transpose(0, 1).to(model.device)
            labels = torch.stack(batch['labels']).transpose(0, 1).to(model.device)
            with torch.no_grad(), autocast('cuda'):
                outputs = model(input_ids=input_ids, labels=labels)
                total_loss += outputs.loss.item()
                total_steps += 1

            if run_exact_match and 'answer' in batch:
                pred_ids = model.generate(input_ids, max_new_tokens=64)
                pred_text = model.config.tokenizer.decode(pred_ids[0], skip_special_tokens=True)
                gold = batch['answer'][0]
                exact_matches += int(_normalize(pred_text) == _normalize(gold))
                total_samples += 1

        avg_loss = total_loss / total_steps if total_steps > 0 else 0.0
        perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')

        result = {'eval_loss': avg_loss, 'perplexity': perplexity}
        if run_exact_match and total_samples > 0:
            result['exact_match'] = exact_matches / total_samples
        return result

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, metrics: dict, round_idx, client_id):
        prefix = ""
        if round_idx is not None:
            prefix += f"Round {round_idx} | "
        if client_id is not None:
            prefix += f"Client {client_id} | "
        parts = [f"{k}: {v:.4f}" for k, v in metrics.items()]
        print(prefix + " | ".join(parts))


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase and strip punctuation/whitespace for exact-match comparison."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return ' '.join(text.split())
