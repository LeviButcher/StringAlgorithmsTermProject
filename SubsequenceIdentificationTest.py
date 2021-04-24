import numpy as np
from cv2 import cv2
import networkx as nx
from networkx.algorithms import bipartite

def get_histograms(frame1, frame2):
    # https://www.researchgate.net/figure/Histogram-Intersection-Similarity-Method-HISM-Histograms-have-eight-bins-and-are_fig3_26815688
    #  https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_begins/py_histogram_begins.html
    
    # Assume grayscale and 256 bins is ok for now....
    
    # Convert images to grayscale
    #print(frame1)
    #print(frame2)
    im1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Calculate histogram for each images
    hist1 = cv2.calcHist([im1], [0], None, [256], [0,256])
    hist2 = cv2.calcHist([im2], [0], None, [256], [0,256])
    
    # Normalize histograms
    max1 = max(hist1)*1.0
    max2 = max(hist2)*1.0
    hist1 = hist1/max1
    hist2 = hist2/max2
    
    """
    # Show histograms
    plt.plot(hist1)
    plt.figure()
    plt.plot(hist2)
    plt.show()
    """
    
    return hist1, hist2

def histogram_intersection(frame1, frame2):
    
    # Get histograms for the images
    hist1, hist2 = get_histograms(frame1, frame2)
    
    # Calculate normalized intersection distance
    intersection = np.sum(np.minimum(hist1,hist2))
    total_area = np.sum(np.maximum(hist1,hist2)) # Area of union of the two histograms
    distance = intersection/total_area # Normalized, identical histograms have distance of 1.0
    
    # Invert distance so that dissimilar images have distance near 1.0, similar images have distance near 0
    distance = 1.0 - distance
    
    return intersection

def createBipartiteGraph(TargetVideo, QueryVideo, threshold, k):
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

        intersection = histogram_intersection(Tframe, Qframe)
        #print(distance)

        if intersection > threshold:
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

def SubsequenceID(TargetVideo, QueryVideo, threshold, maxEditDistance = 0):
    # imports
    import networkx as nx
    from networkx.algorithms import bipartite

    k = 0
    hitPos = []
    hitSize = []
    hitType = []
    #while k < TargetVideo.framecount - QueryVideo.framecount - maxEditDistance:
    check = int(TargetVideo.get(cv2.CAP_PROP_FRAME_COUNT)) - int(QueryVideo.get(cv2.CAP_PROP_FRAME_COUNT)) - maxEditDistance
    qlength = int(QueryVideo.get(cv2.CAP_PROP_FRAME_COUNT))
    tlength = int(TargetVideo.get(cv2.CAP_PROP_FRAME_COUNT))
    while k < int(TargetVideo.get(cv2.CAP_PROP_FRAME_COUNT)) - int(QueryVideo.get(cv2.CAP_PROP_FRAME_COUNT)) - maxEditDistance -2: #Potentially -2 as well?
        print(k, "/", check)
        #construct bipartite graph of query video and target clip and calculate maximum cardinality of graph and the Frames for the graph
        LMCM, MCM, BG = createBipartiteGraph(TargetVideo, QueryVideo, threshold, k)
        #calculate tf (min tframe in MCM) and tl (max tframe in MCM)
        tf = calcTFirstMCM(MCM)
        tl = calcTLastMCM(MCM)

        #calculate DMCM
        DMCM = tl-tf+1

        #calculate hit type
        hit = calcHit(MCM, LMCM, DMCM, maxEditDistance, int(QueryVideo.get(cv2.CAP_PROP_FRAME_COUNT))-1)
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
                k = k + int(QueryVideo.get(cv2.CAP_PROP_FRAME_COUNT) + maxEditDistance)
            else: 
                #k = max(k + query video length - (size of max card match + edit threshold), tf)
                a = k + int(QueryVideo.get(cv2.CAP_PROP_FRAME_COUNT)) - (LMCM + maxEditDistance)
                k = max(a, tf)

    return (hitPos, hitSize, hitType)

target = cv2.VideoCapture('c:/Users/covar/Documents/GitHub/StringAlgorithmsTermProject/TargetVideo.mp4')
query = cv2.VideoCapture('c:/Users/covar/Documents/GitHub/StringAlgorithmsTermProject/QueryVideo.mp4')
hitPos, hitSize, hitType = SubsequenceID(target, query, 0.8, 0)

print("Query video was found at frame(s): ", hitPos)
    