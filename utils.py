import pandas as pd
from custom_types import *
import video_matching
from sklearn.metrics import confusion_matrix


def compute_precision_recall(cm):
    tn, fp, fn, tp = cm.ravel()
    recall = tp / (tp + fp)
    prec = tp / (tp + fn)
    return (recall, prec)


def measure_distance_performance(l: Lambda, d: Delta, distanceFN: DistanceFN, dataset: VideoDataset) -> ConfusionMatrix:
    predictions = [video_matching.findVideoSeq(l, d, distanceFN, target, query)
                   for target, querySet in dataset for query, _ in querySet]

    return generate_confusion_matrix(dataset, predictions)


def generate_confusion_matrix(dataset: VideoDataset, predictions: List[List[int]]) -> ConfusionMatrix:
    y_true = [truth for _, querySet in dataset for _, truth in querySet]
    y_pred = [len(x) > 0 for x in predictions]
    return confusion_matrix(y_true, y_pred)


def measure_all_distance_performance(l: Lambda, d: Delta, distances: List[DistanceFN], dataset: VideoDataset) -> List[ConfusionMatrix]:
    return [measure_distance_performance(l, d, dF, dataset) for dF in distances]


def measure_all_hyperparams_performance(ls: List[Lambda], ds: List[Delta], distanceFN: DistanceFN, dataset: VideoDataset) -> List[Tuple[Lambda, Delta, ConfusionMatrix]]:
    return [(l, d, measure_distance_performance(l, d, distanceFN, dataset))
            for l in ls for d in ds]
