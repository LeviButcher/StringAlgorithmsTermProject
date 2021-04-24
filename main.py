from get_dataset import get_dataset
from utils import compute_precision_recall, measure_all_hyperparams_performance
from distance_fns import histogram_difference, histogram_intersection, histogram_mean_difference
import pandas as pd
from custom_types import *
import sys

datasetPath = str(sys.argv[1])

allLambdas = [.1, .2, .3]
allDeltas = [.1, .2, .3]

distanceFunctions = [("Histogram Differences", histogram_difference),
                     ("Histogram Intersection", histogram_intersection),
                     ("Histogram Mean Difference", histogram_mean_difference)]


def removeNewLine(cm):
    return str(cm).replace('\n', '')


def get_stat_dataframe(distanceName: str, l: Lambda, d: Delta, cm: ConfusionMatrix):
    rec, prec = compute_precision_recall(cm)
    dataGroup = {"DistanceFN": [distanceName], "Lambda": [l], "Delta": [d],
                 "Confusion Matrix": [cm], "Precision": [prec], "Recall": [rec]}

    return pd.DataFrame(data=dataGroup)


def compute_stats(name: str, df: DistanceFN, ls: List[Lambda], ds: List[Delta], dataset: VideoDataset):
    res = measure_all_hyperparams_performance(
        ls, ds, df, dataset)

    return pd.concat([get_stat_dataframe(name, l, d, cm) for l, d, cm in res])


def compute_stats_all_distances(distances: [Tuple[str, DistanceFN]], ls: List[Lambda], ds: List[Delta], dataset: VideoDataset):
    return pd.concat([compute_stats(
        name, df, ls, ds, dataset) for name, df in distanceFunctions])


print(f"Importing Dataset at: {datasetPath}")
dataset = get_dataset(datasetPath)
print(f"Dataset Length: {len(dataset)}")

print(f"Running on Dataset at: ")
results = compute_stats_all_distances(
    distanceFunctions, allLambdas, allDeltas, dataset)
print(results)
