from Models import TinyLlama
import torch
import requests
import io
import gc
import json
import argparse
import random

from metricsCounter import Metrics, MetricsManager


device = "cuda:0" if torch.cuda.is_available() else "cpu"

def getLayerSplit(numOfLayers, numOfHosts):
    start = 0
    splits = []
    # for i in range(numOfLayers):
    #     splits.append(random.randint(0,numOfHosts))
    incement = numOfLayers//numOfHosts
    for i in range(numOfHosts):
        endPos = start + incement
        splits.append((start, min(endPos, numOfLayers)))
        start = endPos
    return splits

class OtherPlace:
    def __init__(self, modelID, splitPosStart, splitPosEnd, url):
        self.modelID = modelID
        self.splitPosStart = splitPosStart
        self.splitPosEnd = splitPosEnd
        self.baseURL = url
        self.initializeDevice()
        self.memoryTracker = MemoryTracker()

    def initializeDevice(self, islast = False):
        url = f"{self.baseURL}/setup"
        data = {
            "modelID": self.modelID,
            "splitPosStart": self.splitPosStart,
            "splitPosEnd": self.splitPosEnd,
            "islast": islast
        }
        result = requests.post(url, json=data)
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
    
    def getServerUsage(self):
        url = f"{self.baseURL}/end"
        response = requests.get(url)
        return response.json()
    

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


def run(prompt, hosts, metricPath):
    modelID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    localUsage = Metrics()
    with MetricsManager(localUsage):
        model = TinyLlama(device)
        layerSplit = getLayerSplit(model.layerCount, len(hosts)+1)
        model.loadModel(True, True, *layerSplit[0])
    localUsage.setFlopProfiler(model.model)
    with MetricsManager(localUsage):
        cloud = None
        if len(hosts) == 1:
            cloudURL = "http://127.0.0.1:8000"
            cloudURL = hosts[0]
            cloud = OtherPlace(model.modelID, layerSplit[1][0], layerSplit[1][1], cloudURL)
        text = "Hello"
        text = f"<|system|>\nYou are a helpful assistant.<|user|>\n{prompt}<|assistant|>\n"
        inputs = model.tokenizer(text, add_special_tokens=True, return_tensors='pt').to(device)
        inputIDs = inputs.input_ids
        maxToken = 100 
        eos = model.tokenizer.eos_token_id
        with torch.no_grad():
            step = 0
            while step<maxToken and inputIDs[0][-1].item()!=eos:
                step+=1
                hiddenStates = model.getInitialHiddenState(inputIDs)
                hiddenStates = model.forward(hiddenStates, device)
                if cloud!=None:
                    hiddenStates = cloud.process(hiddenStates)
                hiddenStates = model.getFinalHiddenStates(hiddenStates)
                logits = model.model.lm_head(hiddenStates[:,-1,:])
                output = torch.argmax(logits, dim=-1).unsqueeze(0)
                inputIDs = torch.cat([inputIDs, output], dim=-1)
        cloudUsage = None
        if cloud!=None:
            cloudUsage = cloud.getServerUsage()
        output = model.tokenizer.decode(inputIDs[0])
    allMetrics = [localUsage.getJson(), cloudUsage]
    allMetrics = [metric for metric in allMetrics if metric is not None]
    Metrics.saveUsageDict(*allMetrics, metricPath=metricPath)
    return output



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Where is Delhi")
    parser.add_argument('--hosts', nargs='+', type=str, default="")
    parser.add_argument('--metricPath', type=str, default="Metrics.json")
    parsedValues = parser.parse_args()
    output = run(parsedValues.prompt, parsedValues.hosts, parsedValues.metricPath)
    # print(output)
    gc.collect()
    torch.cuda.empty_cache()