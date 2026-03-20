import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModelForCausalLM
from huggingface_hub import snapshot_download
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import argparse

def run(user):
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True
    )
    user = "Where is Delhi"
    prompt = f"<|system|>\nYou are a helpful assistant.<|user|>\n{user}<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)

    outputs = model.generate(**inputs)
    text = tokenizer.batch_decode(outputs)
    return text

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Where is Delhi")
    print(run(parser.parse_args().prompt))