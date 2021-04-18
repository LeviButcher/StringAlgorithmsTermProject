from get_dataset import get_dataset
from utils import compute_precision_recall, measure_all_hyperparams_performance
from distance_fns import histogram_difference, histogram_intersection, histogram_mean_difference
import pandas as pd
from custom_types import *


REAL_WORLD_SET_PATH = "./real_world_dataset"
SIMULATE_SET_PATH = "not existing right now"

allLambdas = [.1, .2, .3]
allDeltas = [.1, .2, .3]

distanceFunctions = [("Histogram Differences", histogram_difference),
                     ("Histogram Intersection", histogram_intersection),
                     ("Histogram Mean Difference", histogram_mean_difference)]


def removeNewLine(cm):
    return str(cm).replace('\n', '')


def compute_stats(name: str, df: DistanceFN, ls: List[Lambda], ds: List[Delta], dataset: VideoDataset):
    res = measure_all_hyperparams_performance(
        ls, ds, df, dataset)

    stats = pd.DataFrame()

    for l, d, cm in res:
        rec, prec = compute_precision_recall(cm)
        dataGroup = {"DistanceFN": [name], "Lambda": [l], "Delta": [d],
                     "Confusion Matrix": [cm], "Precision": [prec], "Recall": [rec]}

        record = pd.DataFrame(data=dataGroup)
        stats = stats.append(record)

    return stats


def compute_stats_all_distances(distances: [Tuple[str, DistanceFN]], ls: List[Lambda], ds: List[Delta], dataset: VideoDataset):
    return pd.concat([compute_stats(
        name, df, allLambdas, allDeltas, real_world_datset) for name, df in distanceFunctions])


real_world_datset = get_dataset(REAL_WORLD_SET_PATH)


print("Running Real World Dataset")
real_world_results = compute_stats_all_distances(
    distanceFunctions, allLambdas, allDeltas, real_world_datset)
print(real_world_results)
