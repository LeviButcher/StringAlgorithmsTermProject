# Defines multiple distance functions for finding the distance between two given frames(images)
# distance = DistanceFn(frame1, frame2)
# Distance is in the range [0,1]
# Histograms are on grayscale images with 256 bins

# Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Distance Functions
# histogram intersection (used by the paper)
# histogram/frame difference
# difference of histograms means
# others

def get_histograms(frame1, frame2):
    # https://www.researchgate.net/figure/Histogram-Intersection-Similarity-Method-HISM-Histograms-have-eight-bins-and-are_fig3_26815688
    #  https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_begins/py_histogram_begins.html
    
    # Assume grayscale and 256 bins is ok for now....
    
    # Convert images to grayscale
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
    
    return distance


def histogram_difference(frame1, frame2):
    
    # Get histograms for the images
    hist1, hist2 = get_histograms(frame1, frame2)
    
    # Calculate normalized histogram difference
    difference = np.sum(np.abs(hist1 - hist2))
    max_possible_diff = 256.0 # 256 bins * (1.0 - 0.0)
    distance = difference/max_possible_diff
    
    return distance


def histogram_mean_difference(frame1, frame2):
    
    # Get histograms for the images
    hist1, hist2 = get_histograms(frame1, frame2)
    
    # Calculate histogram means
    mean1 = np.mean(hist1)
    mean2 = np.mean(hist2)
    
    # Calculate normalized mean distance
    difference = np.abs(mean1 - mean2)
    max_possible_diff = 256.0 # histogram means can be from [0.0 - 256.0]
    distance = difference/max_possible_diff
    
    return distance


# Main for testing
if __name__ == "__main__":
    
    im1 = cv2.imread('./sample_images/hist_lowkey.jpg')
    #im1 = cv2.imread('./sample_images/hist_highkey_auto.jpg')
    
    #im2 = cv2.imread('./sample_images/hist_lowkey.jpg')
    #im2 = cv2.imread('./sample_images/hist_lowkey_auto.jpg')
    im2 = cv2.imread('./sample_images/hist_highkey.jpg')
    
    #int_dist = histogram_intersection(im1, im2)
    #int_dist = histogram_difference(im1, im2)
    int_dist = histogram_mean_difference(im1, im2)
    
    print(int_dist)
    
    """
    plt.imshow(im1)
    plt.show()
    plt.imshow(im2)
    plt.show()
    """
    