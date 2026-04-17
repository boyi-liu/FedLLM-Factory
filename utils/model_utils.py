import torch
import os
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, TaskType


os.environ["CUDA_VISIBLE_DEVICES"] = "3"



def load_model(args):
    model_name = args.model
    
    if args.task_type == 'SEQ_CLS':
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            device_map="auto",
            # torch_dtype=torch.float16
        )
    elif args.task_type == 'CAUSAL_LM':
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",  # auto / cuda:0 / cpu ...
            torch_dtype=torch.float16
        )
    return model

def load_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.task_type == 'CAUSAL_LM':
        tokenizer.padding_side = 'left'
    return tokenizer

def load_lora_config(args):
    if args.task_type == 'SEQ_CLS':
        lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["query", "value"],  # BERT applies "query", "value" instead of "q_proj"/"v_proj"
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="SEQ_CLS"
    )
    elif args.task_type == 'CAUSAL_LM':
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "v_proj"]
        )
    return lora_config