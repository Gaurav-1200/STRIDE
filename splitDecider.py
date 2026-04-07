from enum import Enum
import math
import numpy as np

class PowerState(Enum):
    INUSE: 0
    IDLE: 1
    CHARGING: 0.2

class TheSplit:
    def __init__(self, baseData, gpuReward = 0.5, flopReward = 0.5, rewardState = [-0.2, 0, 0.2], rewardBattery = [-0.2, 0, 0.2], rewardWiFi = [-0.2, 0.2]):
        self.baseData = baseData
        self.rewardState = rewardState # In-Use, Idle, Charging
        self.rewardBattery = rewardBattery # < 20, < 50, >70
        self.rewardWiFi = rewardWiFi # On 5G, On WiFi
        self.timeStep = 1
        self.gpuReward = gpuReward
        self.flopsReward = flopReward

    def getScore(self, flops, gpuMemory, state, battery, isWiFi):
        stateReward = self.rewardState[state.value]
        for idx, availBattery in enumerate([0.1, 0.5]):
            if battery < availBattery:
                batteryReward = self.rewardBattery[idx]
        else:
            batteryReward = self.rewardBattery[-1]
        WiFiReward = self.rewardWiFi[isWiFi]
        return (self.flopsReward*(flops/self.baseData.flops) + self.gpuReward*(gpuMemory/self.baseData.gpuMemory))*(1 + stateReward + batteryReward + WiFiReward)
    
    def doTheSplit(self, timeMS, layerData, commMat, exeMat, deviceData, lambda_mem):
        """
        Args:
            layerData: Layer Profiling Data, FLOPs, GPUMemory
            commMat: Matrix containing Download latency from i to > i and upload from i to < i
            exeMat: time taken, gpuMemory, glops to execute layer i on device j
            deviceData: powerState (INUSE, IDLE, CHARGING), battery (percentage), isWifi (bool), GPU memory, device_0 is Server, device_1...N is edge devices
        """
        deviceCount, layerCount, stepCount = len(deviceData), len(layerData), (timeMS//self.timeStep)+1
        DP = np.full((deviceCount, layerCount, stepCount), float('-inf'))
        pathToGlory = np.zeros((deviceCount, layerCount, stepCount), dtype=int)
        for layerIdx, layer in enumerate(layerData):
            for deviceIdx, device in enumerate(deviceData):
                if deviceIdx > 0:
                    currScore = self.getScore(exeMat[layerIdx][deviceIdx].flops, exeMat[layerIdx][deviceIdx].gpuMemory, device.state, device.battery, device.isWiFi)
                    gpuMemoryPenaly = lambda_mem*exeMat[layerIdx][deviceIdx].gpuMemory/device.gpuMemory
                    currScore-=gpuMemoryPenaly
                else:
                    currScore = 0.0

                exeTime = exeMat[layerIdx][deviceIdx].time
                exeSteps = math.ceil(exeTime/self.timeStep)

                if layerIdx == 0:
                    for t in range(exeSteps, stepCount):
                        DP[deviceIdx][layerIdx][t] = currScore
                else:
                    for prevDev in range(deviceCount):
                        # assuming no tensor size changes per layer
                        commTime = commMat[prevDev][deviceIdx]
                        commSteps = math.ceil(commTime/self.timeStep)
                        totalSteps = exeSteps + commSteps

                        for t in range(totalSteps, stepCount):
                            prevT = t - totalSteps
                            score = DP[prevDev][layerIdx-1][prevT] + currScore
                            if score > DP[deviceIdx][layerIdx][t]:
                                DP[deviceIdx][layerIdx][t] = score
                                pathToGlory[deviceIdx][layerIdx][t] = prevDev
        bestDevice = -1
        bestScore = float('-inf')

        for d in range(deviceCount):
            if DP[d][-1][-1] > bestScore:
                bestScore = DP[d][-1][-1]
                bestDevice = d
        if bestScore == float('-inf'):
            return [], bestScore
        
        theSplit = []
        currDevice = bestDevice
        currTime = stepCount - 1
        for layerIdx in range(layerCount-1,0,-1):
            theSplit.append(currDevice)
            prevDevice = pathToGlory[currDevice][layerIdx][currTime]


            execSteps = math.ceil(exeMat[layerIdx][currDevice].time/self.timeStep)
            commSteps = math.ceil(commMat[prevDevice][currDevice]/self.timeStep)

            currTime = currTime - execSteps - commSteps

            currDevice = prevDevice

        theSplit.append(currDevice)
        theSplit.reverse()


        maxMemory = 0
        prevDevice = -1
        for layerIdx, currDevice in enumerate(theSplit):
            if prevDevice == currDevice:
                maxMemory += exeMat[layerIdx][currDevice].gpuMemory
            else:
                maxMemory = exeMat[layerIdx][currDevice].gpuMemory
            if maxMemory > deviceData[currDevice].gpuMemory:
                return [], -1
            prevDevice = currDevice

        return theSplit, bestScore
    

    def splitWithBinarySearch(self, timeMS, layerData, commMat, exeMat, deviceData, delta = 1e-3, iters=50):
        lambdaMin = 0.0
        lambdaMax = 100.0
        bestSplit = None
        bestScore = float('-inf')
        optimalLambda = lambdaMax
        for iter in range(iters):
            mid = (lambdaMin + lambdaMax) / 2.0
            currentSplit, currentScore = self.doTheSplit(timeMS, layerData, commMat, exeMat, deviceData, mid)
            if currentScore == -1:
                lambdaMin = mid
            else:
                bestSplit = currentSplit
                bestScore = currentScore
                optimalLambda = mid
                lambdaMax = mid
            
            if lambdaMax - lambdaMin < delta:
                break
        if bestSplit is None:
            return [], float('-inf')
        return bestSplit, bestScore  





