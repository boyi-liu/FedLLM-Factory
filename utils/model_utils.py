import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType


os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def load_model(args):
    model_name = args.model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # auto / cuda:0 / cpu ...
        torch_dtype=torch.float16
    )
    return model

def load_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    return tokenizer

def load_lora_config(args):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj"]
    )
    return lora_config