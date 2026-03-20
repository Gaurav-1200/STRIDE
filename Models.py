from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import torch.nn as nn
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map
from huggingface_hub import snapshot_download
import os
from safetensors import safe_open
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

class EmptyLayer(nn.Module):
    def forward(self, *args, **kwargs):
        return None

class Model:
    def __init__(self, modelID, prefDevice):
        self.modelID = modelID
        self.device = "cuda:0" if prefDevice=="cuda" and torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(modelID)
        self.config = AutoConfig.from_pretrained(modelID)
        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(self.config)
        self.layerCount = self.config.num_hidden_layers
        self.weightsPath = self.downloadModel()
    
    def loadModelOld(self, isFirst, isLast, startPos, endPos):
        self.layers = self.getLayerList(startPos, endPos)
        deviceMap = infer_auto_device_map(self.model)
        if self.device == 'cpu':
            deviceMap[''] = 'cpu'
        # deviceMap = {}
        # for key in baseMap:
        #     deviceMap[key] = baseMap[key]
        for i in range(self.layerCount):
            if i<startPos or i>=endPos:
                deviceMap[f"{self.name}.{i}"] = "cpu"
        # for i in range(startPos,endPos):
        #     deviceMap[f"{self.name}.{i}"] = self.device
        # if isFirst:
        #     deviceMap["model.embed_tokens"] = self.device
        # if isLast:
        #     deviceMap["model.norm"] = self.device
        #     deviceMap["lm_head"] = self.device
        # if hasattr(self.model.model, "rotary_emb"):
        #     dim = self.config.hidden_size // self.config.num_attention_heads
        #     # Llama's exact math formula for inv_freq
        #     self.model.model.rotary_emb.inv_freq = 1.0 / (
        #         10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        #     )
        # if "model.rotary_emb" in baseMap:
        #     deviceMap["model.rotary_emb"] = self.device
        self.model = load_checkpoint_and_dispatch(self.model, checkpoint=self.weightsPath, offload_folder="offload", device_map=deviceMap)
    

    def downloadModel(self):
        access_token = ""
        return snapshot_download(repo_id=self.modelID, token=access_token)

    def getLayerList(self, startPos, endPos):
        layerList = None
        for name, module in self.model.named_modules():
            if isinstance(module, nn.ModuleList):
                layerList = module
                self.name = name
                break
        
        # for i in range(len(layerList)):
        #     if i<startPos or i>=endPos:
        #         layerList[i] = EmptyLayer()
        return layerList[startPos:endPos]


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

class EmptyLayer(nn.Module):
    def forward(self, *args, **kwargs):
        return None
    


class TinyLlama(Model):
    def __init__(self, prefDevice):
        self.modelID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        super().__init__(self.modelID, prefDevice)
    def getInitialHiddenState(self, inputIDs):
        hiddenStates = self.model.model.embed_tokens(inputIDs)
        return hiddenStates

    def materializeRotaryEmbeddings(self):
        if hasattr(self.model.model, "rotary_emb"):
            dim = self.config.hidden_size // self.config.num_attention_heads
            # Llama's exact math formula for inv_freq
            # self.model.model.rotary_emb.inv_freq = 1.0 / (
            #     10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
            # ).to(self.device)
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


    def forward(self):
        pass
    
    def getFinalHiddenStates(self, hiddenStates):
        return self.model.model.norm(hiddenStates)