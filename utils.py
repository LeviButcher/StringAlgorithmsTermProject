import pandas as pd
from custom_types import *
import video_matching
from sklearn.metrics import confusion_matrix
import get_dataset


def compute_precision_recall(cm):
    tn, fp, fn, tp = cm.ravel()
    recall = tp / (tp + fp)
    prec = tp / (tp + fn)
    return (recall, prec)


def evaluate_target_set(l: Lambda, d: Delta, distanceFN: DistanceFN, target: str, queries: [str]):
    targetVideo = get_dataset.get_video_frame(target)
    queryVideos = [get_dataset.get_video_frame(q) for q, _ in queries]
    return [video_matching.findVideoSeq(l, d, distanceFN, targetVideo, q) for q in queryVideos]


def measure_distance_performance(l: Lambda, d: Delta, distanceFN: DistanceFN, dataset: VideoDataset) -> ConfusionMatrix:
    predictions = [evaluate_target_set(l, d, distanceFN, target, querySet)
                   for target, querySet in dataset]

    return generate_confusion_matrix(dataset, predictions)


def generate_confusion_matrix(dataset: VideoDataset, predictions: List[List[List[int]]]) -> ConfusionMatrix:
    y_true = [truth for _, querySet in dataset for _, truth in querySet]
    y_pred = [len(xx) > 0 for x in predictions for xx in x]
    return confusion_matrix(y_true, y_pred)


def measure_all_distance_performance(l: Lambda, d: Delta, distances: List[DistanceFN], dataset: VideoDataset) -> List[ConfusionMatrix]:
    return [measure_distance_performance(l, d, dF, dataset) for dF in distances]


def measure_all_hyperparams_performance(ls: List[Lambda], ds: List[Delta], distanceFN: DistanceFN, dataset: VideoDataset) -> List[Tuple[Lambda, Delta, ConfusionMatrix]]:
    return [(l, d, measure_distance_performance(l, d, distanceFN, dataset))
            for l in ls for d in ds]


def removeNewLine(cm: ConfusionMatrix):
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
        name, df, ls, ds, dataset) for name, df in distances])
