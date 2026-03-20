from Models import TinyLlama
import torch
import requests
import io
import gc
import json
import argparse


device = "cuda:0" if torch.cuda.is_available() else "cpu"

def getLayerSplit(numOfLayers, numOfHosts):
    print(numOfLayers)
    start = 0
    splits = []
    incement = numOfLayers//numOfHosts
    for i in range(numOfHosts):
        endPos = start + incement
        splits.append((start, min(endPos, numOfLayers)))
        start = endPos+1
    return splits

class OtherPlace:
    def __init__(self, modelID, splitPosStart, splitPosEnd, url):
        self.modelID = modelID
        self.splitPosStart = splitPosStart
        self.splitPosEnd = splitPosEnd
        self.baseURL = url
        self.initializeDevice()
        self.memoryTracker = MemoryTracker()

    def initializeDevice(self):
        url = f"{self.baseURL}/setup"
        data = {
            "modelID": self.modelID,
            "splitPosStart": self.splitPosStart,
            "splitPosEnd": self.splitPosEnd
        }
        result = requests.post(url, json=data)
        print(result.content)
        if not result.json()["status"]:
            raise Exception("Not Initialized")

    def process(self, hiddenStates):
        url = f"{self.baseURL}/process"
        buffer = io.BytesIO()
        torch.save(hiddenStates.cpu(), buffer)
        dataToSend = buffer.getvalue()
        self.memoryTracker.add(len(dataToSend)/(1024*1024), "Sent")
        response = requests.post(url, data=dataToSend)
        responseBuffer = io.BytesIO(response.content)
        outputFromServer = torch.load(responseBuffer, weights_only=True)
        self.memoryTracker.add(len(response.content)/(1024*1024), "Received")
        return outputFromServer.to(device)
    

class MemoryTracker:
    def __init__(self):
        self.filePath = "Memory.json"
        with open("Memory.json", "w") as f:
            f.write(json.dumps({}))

    def add(self, dataSize, dataName):
        with open("Memory.json") as f:
            oldData = json.loads(f.read())
        if dataName not in oldData:
            oldData[dataName] = []
        oldData[dataName].append(dataSize)
        with open("Memory.json", "w") as f:
            f.write(json.dumps(oldData))

    @staticmethod
    def getSummary(filePath):
        with open(filePath) as f:
            oldData = json.loads(f.read())
        summary = {data: sum(oldData[data]) for data in oldData}
        return summary


def run(prompt, hosts):
    modelID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model = TinyLlama(device)
    layerSplit = getLayerSplit(model.layerCount, len(hosts)+1)
    cloud = None
    if len(hosts) == 1:
        cloudURL = "http://127.0.0.1:8000"
        cloudURL = hosts[0]
        cloud = OtherPlace(model.modelID, layerSplit[1][0], layerSplit[1][1], cloudURL)
    model.loadModel(True, True, *layerSplit[0])
    text = "Hello"
    text = f"<|system|>\nYou are a helpful assistant.<|user|>\n{prompt}<|assistant|>\n"
    # inputs = model.tokenizer(text, return_tensors="pt")
    inputs = model.tokenizer(text, add_special_tokens=True, return_tensors='pt').to(device)
    inputIDs = inputs.input_ids
    maxToken = 100 
    eos = model.tokenizer.eos_token_id
    with torch.no_grad():
        step = 0
        while step<maxToken and inputIDs[0][-1].item()!=eos:
            step+=1
            
            hiddenStates = model.getInitialHiddenState(inputIDs)
            seq_length = inputIDs.shape[1]
            position_ids = torch.arange(0, seq_length, dtype=torch.long).unsqueeze(0)
            rotary_module = model.model.model.rotary_emb
            position_embeddings = rotary_module(hiddenStates, position_ids.to(device))
            attention_mask = torch.tril(torch.ones((1, 1, seq_length, seq_length),device=device))
            attention_mask = (1.0 - attention_mask) * torch.finfo(hiddenStates.dtype).min
            attention_mask = attention_mask.to(hiddenStates.dtype)
            for layer in model.layers:
                outputs = layer(hiddenStates,attention_mask=attention_mask,
            position_ids=position_ids,position_embeddings=position_embeddings,
            use_cache=False)
                hiddenStates = outputs
            if cloud!=None:
                hiddenStates = cloud.process(hiddenStates)
            hiddenStates = model.getFinalHiddenStates(hiddenStates)
            logits = model.model.lm_head(hiddenStates)
            output = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
            inputIDs = torch.cat([inputIDs, output], dim=-1)
    return model.tokenizer.decode(inputIDs[0])



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Where is Delhi")
    parser.add_argument('--hosts', nargs='+', type=str, default="")
    parsedValues = parser.parse_args()
    print(run(parsedValues.prompt, parsedValues.hosts))
    gc.collect()
    torch.cuda.empty_cache()