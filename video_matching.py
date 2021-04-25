from custom_types import *
from get_dataset import get_video_frame

# Graph comparison on target and query video
def createBipartiteGraph(TargetVideo, QueryVideo, threshold, k, DistanceFN):
    from distance_fns import histogram_intersection
    import networkx as nx
    from networkx.algorithms import bipartite

    #print("graph")
    BG = nx.Graph()
    qNodes = []
    tNodes = []
    TargetVideo.set(2,0)
    QueryVideo.set(2,0)

    for i in range(0,k-1): 
        successT, Tframe = TargetVideo.read()

    for i in range(0,int(QueryVideo.get(cv2.CAP_PROP_FRAME_COUNT))-1):
        successT, Tframe = TargetVideo.read()
        #BG.add_nodes_from([k+i], bipartite=0)
        tNodes.append(k+i)
        successQ, Qframe = QueryVideo.read()
        #BG.add_nodes_from([i], bipartite=1)
        qNodes.append(str(i))

        intersection = DistanceFN(Tframe, Qframe)
        #print(distance)

        if intersection < threshold:
            BG.add_edges_from([(k+i,str(i))])
    
    BG.add_nodes_from(tNodes, bipartite=0)
    BG.add_nodes_from(qNodes, bipartite=1)
    #print(BG.nodes)
    #print(BG.edges)
    MCM = bipartite.matching.maximum_matching(BG, qNodes)
    LMCM = int(len(MCM)/2)
    #print(MCM)
    return LMCM, MCM, BG

def calcTFirstMCM(MCM):
    #min of t frames in MCM
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
    if LMCM == M and DMCM == LMCM:                                                                     #if size of MCM == length of query and DMCM = size of MCM; hit == 1
        hit = 1
    elif LMCM == M and LMCM < DMCM and DMCM <= M + maxEditDistance:                                    #if size of MCM == length of query and size of MCM < DMCM <= size of query + max edit distance; hit == 2
        hit = 2
    elif M - maxEditDistance <= LMCM and LMCM < M and DMCM == LMCM:                                #if length of query - max edit distance <= size of MCM < length of query and DMCM == size of MCM; hit == 3
        hit = 3
    elif M - maxEditDistance <= LMCM and LMCM < M and DMCM == M:                                       #if length of query - max edit distance <= size of MCM < length of query and DMCM == length of query; hit == 4
        hit = 4
    elif M - maxEditDistance <= LMCM and LMCM < M and LMCM <= DMCM and DMCM < M:                   #if length of query - max edit distance <= size of MCM < length of query and size of MCM <= DMCM < length of query; hit == 5
        hit = 5
    elif M - maxEditDistance <= LMCM and LMCM < M and M < DMCM and DMCM <= LMCM + maxEditDistance: #if length of query - max edit distance <= size of MCM < length of query and length of query < DMCM <= size of MCM + max edit distance; hit == 6
        hit = 6
    else:
        hit = 0

    return hit

def findVideoSeq(maxEditDistance: Lambda, threshold: Delta, get_distance: DistanceFN, target: Video, query: Video) -> List[int]:
    # imports
    import networkx as nx
    from networkx.algorithms import bipartite
    from cv2 import cv2

    target = get_video_frame('TargetVideo.mp4')
    query = get_video_frame('QueryVideo.mp4')

    k = 0
    hitPos = []
    hitSize = []
    hitType = []
    #while k < TargetVideo.framecount - QueryVideo.framecount - maxEditDistance:
    check = int(target.get(cv2.CAP_PROP_FRAME_COUNT)) - int(query.get(cv2.CAP_PROP_FRAME_COUNT)) - maxEditDistance
    while k < check-2: #Potentially -2 as well?
        print(k, "/", check)
        #construct bipartite graph of query video and target clip and calculate maximum cardinality of graph and the Frames for the graph
        LMCM, MCM, BG = createBipartiteGraph(target, query, threshold, k, DistanceFN)
        #calculate tf (min tframe in MCM) and tl (max tframe in MCM)
        tf = calcTFirstMCM(MCM)
        tl = calcTLastMCM(MCM)

        #calculate DMCM
        DMCM = tl-tf+1

        #calculate hit type
        hit = calcHit(MCM, LMCM, DMCM, maxEditDistance, int(query.get(cv2.CAP_PROP_FRAME_COUNT))-1)
        #if hit function does not return 0
        if hit > 0: #Query was found at location k
            hitPos.append(tf)
            hitSize.append(DMCM)
            hitType.append(hit)
            k = tl + 1
        else: #else query not found
            #if size of max card match = 0 then
            if LMCM == 0:
                #k=k+(length of Query video + max edit distance)
                k = k + int(query.get(cv2.CAP_PROP_FRAME_COUNT) + maxEditDistance)
            else: 
                #k = max(k + query video length - (size of max card match + edit threshold), tf)
                a = k + int(query.get(cv2.CAP_PROP_FRAME_COUNT)) - (LMCM + maxEditDistance)
                k = max(a, tf)
    return []

#note, when specifying the threshold, as of right now it is based on distance and not similarity, i.e. if the distance is < the threshold it is a hit
# I was having trouble with get_video_frame