import os
import json
import numpy as np
import pandas as pd
from metricsCounter import MetricConstants
import argparse
def saveJson(path, data):
    with open(path,"w") as f:
        json.dump(data, f)
def loadJson(path):
    with open(path) as f:
        data = json.load(f)
    return data
def saveMetrics(baseCase, splitCase, prompt, fileName = "FullComparison.csv"):
    baseCaseDevice = list(baseCase.keys())[0]
    splitCaseName = f"Split: {len(splitCase)}"
    header = ["Prompt", "Metric", baseCaseDevice, splitCaseName, "Difference", "% Decrease"]
    metricsComparison = []
    for metricToMeasure in MetricConstants:
        value = metricToMeasure.value
        comparison = [prompt, value]
        comparison.append(baseCase[baseCaseDevice][value])
        metricValue = 0
        for device in splitCase.keys():
            metricValue = max(metricValue, splitCase[device][value])
        comparison.append(metricValue)
        comparison.append(comparison[2] - comparison[-1])
        comparison.append((comparison[-1]/comparison[2])*100)
        
        for idx,value in enumerate(comparison):
            try:
                comparison[idx] = round(value, 2)
            except:
                pass
        metricsComparison.append(comparison)
    if os.path.exists(fileName):
        oldData = pd.read_csv(fileName)
        dataframe = pd.DataFrame(metricsComparison, columns=header)
        dataframe = pd.concat([oldData, dataframe])
    else:
        dataframe = pd.DataFrame(metricsComparison, columns=header)
    dataframe.to_csv(fileName, index=False, header=True)
    return dataframe
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Where is Delhi")
    parser.add_argument("--promptJson", type=str)
    parser.add_argument('--hosts', nargs='+', type=str, required=True)
    parser.add_argument('--comparisonPath', nargs='+', type=str, default="FullComparison.csv")
    parsedValues = parser.parse_args()
    comparisonPath = parsedValues.comparisonPath
    if parsedValues.promptJson:
        prompts = loadJson(parsedValues.promptJson)
    else:
        prompts = [parsedValues.prompt]
    for idx, prompt in enumerate(prompts):
        # base case
        metricPath = "Metrics_base.json"
        os.system(f'python3 theSplit.py --prompt "{prompt}" --metricPath {metricPath}')
        metrics_base = loadJson(metricPath)

        # 1-split
        metricPath = "Metrics_split.json"
        os.system(f'python3 theSplit.py --prompt "{prompt}" --hosts {' '.join(parsedValues.hosts)} --metricPath {metricPath}')
        metrics_split = loadJson(metricPath)

        # comparison
        comparison = saveMetrics(metrics_base, metrics_split, prompt, comparisonPath)