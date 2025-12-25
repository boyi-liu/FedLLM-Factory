import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel


os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def load_model(args):
    model_name = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # auto / cuda:0 / cpu ...
        torch_dtype=torch.float16
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj"]
    )

    return get_peft_model(model, lora_config), tokenizer