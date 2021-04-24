from get_dataset import get_dataset
from utils import compute_stats_all_distances
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


print(f"Importing Dataset at: {datasetPath}")
dataset = get_dataset(datasetPath)
print(f"Dataset Length: {len(dataset)}")

print(f"Running on Dataset at: {datasetPath}")
results = compute_stats_all_distances(
    distanceFunctions, allLambdas, allDeltas, dataset)
print(results)

results.to_csv("results.csv")
print("Results written to: results.csv")
