from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import torch.nn as nn
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map
from huggingface_hub import snapshot_download
import os
from safetensors import safe_open
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
import config


class Model:
    def __init__(self, modelID, prefDevice):
        self.modelID = modelID
        self.device = prefDevice if "cuda" in prefDevice and torch.cuda.is_available() else "cpu"
        self.weightsPath = self.downloadModel()
        self.tokenizer = AutoTokenizer.from_pretrained(modelID, local_files_only=True)
        self.config = AutoConfig.from_pretrained(modelID, local_files_only=True)
        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(self.config)
        self.layerCount = self.config.num_hidden_layers
    

    def downloadModel(self):
        access_token = config.accessToken
        try:
            return snapshot_download(repo_id=self.modelID, token=access_token, local_files_only=True)
        except:
            return snapshot_download(repo_id=self.modelID, token=access_token)


class GPT2(Model):
    def __init__(self, prefDevice):
        self.modelID = "gpt2"
        super().__init__(self.modelID, prefDevice)
    def getInitialHiddenState(self, inputIDs):
        positionIDs = torch.arange(0, inputIDs.size(-1), dtype=torch.long, device=inputIDs.device)
        positionIDs = positionIDs.unsqueeze(0)
        inputEmbeds = self.model.transformer.wte(inputIDs)
        positionEmbeds = self.model.transformer.wpe(positionIDs)
        hiddenStates = inputEmbeds + positionEmbeds
        return hiddenStates

    def forward(self):
        pass
    
    def getFinalHiddenStates(self, hiddenStates):
        return self.model.transformer.ln_f(hiddenStates)
    


class TinyLlama(Model):
    def __init__(self, prefDevice):
        self.modelID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        super().__init__(self.modelID, prefDevice)
    def getInitialHiddenState(self, inputIDs):
        hiddenStates = self.model.model.embed_tokens(inputIDs)
        return hiddenStates

    def materializeRotaryEmbeddings(self):
        if hasattr(self.model.model, "rotary_emb"):
            if not hasattr(self.config, "rope_theta"):
                self.config.rope_theta = 10000.0
            self.model.model.rotary_emb = LlamaRotaryEmbedding(
                config=self.config,
                device=self.device,
            )

    def loadModel(self, isFirst, isLast, startPos, endPos):
        self.model.model.layers = nn.ModuleList(self.model.model.layers[startPos:endPos])
        self.layers = self.model.model.layers
        self.model.to_empty(device=self.device)
        self.materializeRotaryEmbeddings()
        stateDict = self.model.state_dict()
        with safe_open(os.path.join(self.weightsPath, "model.safetensors"), framework="pt", device="cpu") as file:
            for key in file.keys():
                if "model.layers." in key:
                    layerIdx = int(key.split(".")[2])
                    if startPos <= layerIdx < endPos:
                        localLayerIdx = layerIdx-startPos
                        newKey = key.replace(f"layers.{layerIdx}", f"layers.{localLayerIdx}")
                        if newKey in stateDict:
                            stateDict[newKey].copy_(file.get_tensor(key).to(self.device, non_blocking=True))
                elif "embed_tokens" in key or "lm_head" in key or "model.norm" in key:
                    if key in stateDict:
                        stateDict[key].copy_(file.get_tensor(key).to(self.device, non_blocking=True))
        self.model.eval()


    def forward(self, hiddenStates, device):
        seq_length = hiddenStates.shape[1]
        position_ids = torch.arange(0, seq_length, dtype=torch.long).unsqueeze(0)
        rotary_module = self.model.model.rotary_emb
        position_embeddings = rotary_module(hiddenStates, position_ids.to(device))
        attention_mask = torch.tril(torch.ones((1, 1, seq_length, seq_length),device=device))
        attention_mask = (1.0 - attention_mask) * torch.finfo(hiddenStates.dtype).min
        attention_mask = attention_mask.to(hiddenStates.dtype)
        for layer in self.layers:
            hiddenStates = layer(hiddenStates,
                            attention_mask=attention_mask, 
                            position_ids=position_ids,
                            position_embeddings=position_embeddings,
                            use_cache=False)
        return hiddenStates
    
    def getFinalHiddenStates(self, hiddenStates):
        return self.model.model.norm(hiddenStates)