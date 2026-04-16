import os
import json
import numpy as np
import pandas as pd
from metricsCounter import MetricConstants
import argparse
def saveJson(path, data):
    with open(path,"w") as f:
        json.dump(data, f, indent=2)
def loadJson(path):
    with open(path) as f:
        data = json.load(f)
    return data
def saveMetrics(baseCase, splitCase, prompt, fileName = "FullComparison.csv"):
    baseCaseDevice = list(baseCase.keys())[0]
    splitCaseName = f"Split: {len(splitCase)}"
    header = ["Prompt", "Metric", baseCaseDevice, splitCaseName, "Difference", "% Decrease"]
    metricsComparison = []
    jsonComparisonData = {
        "prompt": prompt
    }
    for metricToMeasure in MetricConstants:
        value = metricToMeasure.value
        comparison = [prompt, value]
        comparison.append(baseCase[baseCaseDevice][value])
        if metricToMeasure == MetricConstants.TimeTaken:
            metricValue = splitCase["Local"][value]
        else:
            metricValue = splitCase["Server"][value]
        # metricValue = 0
        # for device in splitCase.keys():
        #     metricValue = max(metricValue, splitCase[device][value])
        comparison.append(metricValue)
        difference = comparison[2] - comparison[-1]
        comparison.append(difference)
        comparison.append((comparison[-1]/comparison[2])*100)
        jsonComparisonData[value] = [baseCase[baseCaseDevice][value], metricValue, difference]
        
        for idx,value in enumerate(comparison):
            try:
                comparison[idx] = round(value, 2)
            except:
                pass
        metricsComparison.append(comparison)
    if os.path.exists(fileName):
        oldData = pd.read_csv(fileName)
        jsonData = loadJson(fileName.replace(".csv",".json"))
        dataframe = pd.DataFrame(metricsComparison, columns=header)
        dataframe = pd.concat([oldData, dataframe])
        jsonData.append(jsonComparisonData)
    else:
        dataframe = pd.DataFrame(metricsComparison, columns=header)
        jsonData = [jsonComparisonData]
    dataframe.to_csv(fileName, index=False, header=True)
    saveJson(fileName.replace(".csv",".json"), jsonData)
    return dataframe


def comparisonSummary(completeData, fileToSave):
    data = {
        "Split": {},
        "Local": {},
        "Gain": {}
    }
    for metricToMeasure in MetricConstants:
        if metricToMeasure == MetricConstants.PEAKVRAM:
            data["Split"][metricToMeasure.value] = max([value[metricToMeasure.value][1] for value in completeData])
        else:
            data["Split"][metricToMeasure.value] = np.mean([value[metricToMeasure.value][1] for value in completeData])
        if metricToMeasure == MetricConstants.PEAKVRAM:
            data["Local"][metricToMeasure.value] = max([value[metricToMeasure.value][0] for value in completeData])
        else:
            data["Local"][metricToMeasure.value] = np.mean([value[metricToMeasure.value][0] for value in completeData])
        data["Gain"][metricToMeasure.value] = 100*(data["Local"][metricToMeasure.value]-data["Split"][metricToMeasure.value])/data["Local"][metricToMeasure.value]
    saveJson(fileToSave, data)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Where is Delhi")
    parser.add_argument("--promptJson", type=str)
    parser.add_argument('--hosts', nargs='+', type=str, required=True)
    parser.add_argument('--comparisonPath', nargs='+', type=str, default="FullComparison.csv")
    parser.add_argument('--resetComparison', action='store_true')
    parsedValues = parser.parse_args()
    jsonPath = parsedValues.comparisonPath.replace(".csv",".json")
    if parsedValues.resetComparison:
        os.remove(parsedValues.comparisonPath)
        os.remove(jsonPath)
    comparisonPath = parsedValues.comparisonPath
    if parsedValues.promptJson:
        prompts = loadJson(parsedValues.promptJson)
    else:
        prompts = [parsedValues.prompt]*4
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
    completeData = loadJson(jsonPath)
    comparisonSummary(completeData, "Comparison_All.json")
    if len(completeData) > 100:
        comparisonSummary(completeData[5:], "Comparison_First5.json")
        comparisonSummary(completeData[10:], "Comparison_First10.json")
        comparisonSummary(completeData[50:], "Comparison_First50.json")


        