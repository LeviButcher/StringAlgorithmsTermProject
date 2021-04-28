from custom_types import *
from get_dataset import get_video_frame
from distance_fns import *

# Graph comparison on target and query video


def createBipartiteGraph(TargetVideo, QueryVideo, threshold, k, DistanceFN):
    from distance_fns import histogram_intersection
    import networkx as nx
    from networkx.algorithms import bipartite

    # print("graph")
    BG = nx.Graph()
    qNodes = []
    tNodes = []

    # for i in range(0,k-1):
    #    successT, Tframe = TargetVideo.read()

    for i in range(0, len(QueryVideo)):
        # successT, Tframe = TargetVideo.read()
        # BG.add_nodes_from([k+i], bipartite=0)
        tNodes.append(k+i)
        # successQ, Qframe = QueryVideo.read()
        # BG.add_nodes_from([i], bipartite=1)
        qNodes.append(str(i))

        tframe = TargetVideo[k+i]
        qframe = QueryVideo[i]
        intersection = DistanceFN(TargetVideo[k+i], QueryVideo[i])
        # print(distance)

        if intersection < threshold:
            BG.add_edges_from([(k+i, str(i))])

    BG.add_nodes_from(tNodes, bipartite=0)
    BG.add_nodes_from(qNodes, bipartite=1)
    # print(BG.nodes)
    # print(BG.edges)
    MCM = bipartite.matching.maximum_matching(BG, qNodes)
    LMCM = int(len(MCM)/2)
    # print(MCM)
    return LMCM, MCM, BG


def calcTFirstMCM(MCM):
    # min of t frames in MCM
    tkeys = [num for num in MCM.keys() if isinstance(num, (int))]
    tvals = [num for num in MCM.values() if isinstance(num, (int))]
    if len(tkeys) == 0 or len(tvals) == 0:
        minimum = 0
    else:
        minimum = min(min(tkeys), min(tvals))
    return minimum


def calcTLastMCM(MCM):
    tkeys = [num for num in MCM.keys() if isinstance(num, (int))]
    tvals = [num for num in MCM.values() if isinstance(num, (int))]
    if len(tkeys) == 0 or len(tvals) == 0:
        maximum = 0
    else:
        maximum = max(max(tkeys), max(tvals))
    return maximum


def calcHit(MCM, LMCM, DMCM, maxEditDistance, M):
    hit = 0
    if LMCM == M and DMCM == LMCM:  # if size of MCM == length of query and DMCM = size of MCM; hit == 1
        hit = 1
    # if size of MCM == length of query and size of MCM < DMCM <= size of query + max edit distance; hit == 2
    elif LMCM == M and LMCM < DMCM and DMCM <= M + maxEditDistance:
        hit = 2
    # if length of query - max edit distance <= size of MCM < length of query and DMCM == size of MCM; hit == 3
    elif M - maxEditDistance <= LMCM and LMCM < M and DMCM == LMCM:
        hit = 3
    # if length of query - max edit distance <= size of MCM < length of query and DMCM == length of query; hit == 4
    elif M - maxEditDistance <= LMCM and LMCM < M and DMCM == M:
        hit = 4
    # if length of query - max edit distance <= size of MCM < length of query and size of MCM <= DMCM < length of query; hit == 5
    elif M - maxEditDistance <= LMCM and LMCM < M and LMCM <= DMCM and DMCM < M:
        hit = 5
    # if length of query - max edit distance <= size of MCM < length of query and length of query < DMCM <= size of MCM + max edit distance; hit == 6
    elif M - maxEditDistance <= LMCM and LMCM < M and M < DMCM and DMCM <= LMCM + maxEditDistance:
        hit = 6
    else:
        hit = 0

    return hit


def findVideoSeq(l: Lambda, threshold: Delta, get_distance: DistanceFN, target: Video, query: Video) -> List[List[int]]:
    from networkx.algorithms import bipartite
    import math

    N = len(target)
    M = len(query)
    maxEditDistance = math.floor(l * M)

    k = 0
    hitPos = []
    hitSize = []
    hitType = []

    print(N, M)

    while k < N - M - maxEditDistance:
        print(k, "/", N - M - maxEditDistance)

        LMCM, MCM, BG = createBipartiteGraph(
            target, query, threshold, k, get_distance)

        tf = calcTFirstMCM(MCM)
        tl = calcTLastMCM(MCM)

        DMCM = tl-tf+1

        hit = calcHit(MCM, LMCM, DMCM, maxEditDistance, M)
        if hit > 0:  # Query was found at location k
            hitPos.append(tf)
            hitSize.append(DMCM)
            hitType.append(hit)
            k = tl + 1
        else:  # else query not found
            # if size of max card match = 0 then
            if LMCM == 0:
                # k=k+(length of Query video + max edit distance)
                k = k + M + maxEditDistance
            else:
                # k = max(k + query video length - (size of max card match + edit threshold), tf)
                a = k + M - (LMCM + maxEditDistance)

                k = max(a, tf)

    return hitPos


# note, when specifying the threshold, as of right now it is based on distance and not similarity, i.e. if the distance is < the threshold it is a hit
# I was having trouble with get_video_frame
# TargetVideo = get_video_frame('../../Downloads/SimulatedDataset/query1.mp4')
# QueryVideo = get_video_frame('c:/Users/Emmy/Downloads/query1.mp4')

# one, two = get_histograms(TargetVideo[0], QueryVideo[0])
# print("done")
