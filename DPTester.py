import json
import os
from enum import Enum
import argparse
 
import numpy as np
from splitDecider import TheDecider
class PowerState(Enum):
    INUSE = 0
    IDLE = 1
    CHARGING = 2
 
 
 
class Layer:
    def __init__(self, memory):
        self.memory = memory
 
class Execution:
    def __init__(self, timeTaken, gpuMemory, flops):
        self.time = timeTaken
        self.gpuMemory = gpuMemory
        self.flops = flops
class Device:
    def __init__(self, powerState, battery, isWifi, gpuMemory):
        self.state = powerState
        self.battery = battery
        self.isWiFi = isWifi
        self.gpuMemory = gpuMemory
 
 
def loadJson(path):
    with open(path) as f:
        data = json.loads(f.read())
    return data
 
def test(sla_latency):
    pathServer = os.path.join("results", "split_bert_layer12_server.json")
    pathClient = os.path.join("results", "split_bert_layer12_client.json")
    pathBase = os.path.join("results", "baseline_bert.json")
    serverData = loadJson(pathServer)
    clientData = loadJson(pathClient)
    clientData2 = loadJson(pathClient)
    baseData = loadJson(pathBase)
    layerCount = 12
    deviceCount = 3
    layerData = [1000 for _ in range(layerCount)]
    commMat = np.full((deviceCount, deviceCount), 0.02)
    exeMat = np.zeros((layerCount, deviceCount), dtype=Execution)
    # deviceData = np.zeros(deviceCount)
    deviceData = [
        Device(PowerState.IDLE, 0.4, True, 50),
        Device(PowerState.IDLE, 0.4, True, 1),
        Device(PowerState.CHARGING, 0.4, True, 1)
    ]
    for deviceIdx, device in enumerate((serverData, clientData, clientData2)):
        for layerIdx in range(layerCount):
            for data in device["layer_metrics"]:
                if data["layer_name"] == f"bert.encoder.layer.{layerIdx}":
                    execution = Execution(data["latency_ms"]*(deviceIdx+1)**2, data["mem_delta_mb"], data["flops"])
                    exeMat[layerIdx][deviceIdx] = execution
    baseline = Execution(0, baseData["total_flops"], baseData["total_mem_mb"])
    theSplit = TheDecider(baseline)
    result = theSplit.splitWithBinarySearch(sla_latency, layerData, commMat, exeMat, deviceData)
    # print(result)
 
 
 
parser = argparse.ArgumentParser()
parser.add_argument("--sla_latency", type=int, default=200)
parsedValues = parser.parse_args()
test(parsedValues.sla_latency)