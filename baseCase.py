import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModelForCausalLM
from huggingface_hub import snapshot_download
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True
)
prompt = 'Where is Delhi'
inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)

outputs = model.generate(**inputs)
text = tokenizer.batch_decode(outputs)
print(text)