from typing import Tuple, List, Callable

# Types needed for various functions

Lambda = float
Delta = float
Precision = float
Recall = float
Frame = List[List[List[int]]]  # n x m x 3 dims
Video = List[Frame]  # t length video
VideoDataset = List[str, List[Tuple[str, bool]]]]
ConfusionMatrix = List[List[int]]
DistanceFN = Callable[[Frame, Frame], float]
