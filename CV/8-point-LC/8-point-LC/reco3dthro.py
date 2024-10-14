# part of the code loosely based on example from https://docs.opencv.org/4.10.0/da/de9/tutorial_py_epipolar_geometry.html
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# draw pre-calculated epipolar lines into images
def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)

    x = int(len(lines) / 20)  # plot only 20 lines
    i = 0
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        i = i+1
        if (i % x == 0):
            # color = tuple(np.random.randint(0,255,3).tolist())
            color = (255, 0, 0)
            x0,y0 = map(int, [0, -r[2]/r[1] ])
            x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
            img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
            img1 = cv.circle(img1,tuple(np.int32(pt1)),5,color,-1)
            img2 = cv.circle(img2,tuple(np.int32(pt2)),5,color,-1)

    return img1,img2


# Find epipolar lines corresponding to points in right image (second image) and
# draw the lines on left image
def drawEpipolarLines(img1, img2, pts1, pts2, F):
    lines = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines = lines.reshape(-1, 3)
    eimg1, eimg2 = drawlines(img1, img2, lines, pts1, pts2)
    return eimg1


# find left/right correspondences
def findpointmatches(img1, img2, removeoutliers=True):
    sift = cv.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    pts1 = []
    pts2 = []

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.float64(pts1)
    pts2 = np.float64(pts2)

    # this is just a quick & dirty way to remove outliers
    # of course, in practice we would not compute the F matrix afterwards again...
    # this is just because we have not discussed outlier removal yet
    # try turning it off...
    if removeoutliers:
        F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)

        # We select only inlier points
        pts1 = pts1[mask.ravel() == 1]
        pts2 = pts2[mask.ravel() == 1]

    return pts1, pts2


# read calibration data txt file and return intrinsic matrices
def readCalibFile(filename):
    file = open(filename, "r")
    line1 = file.readline()
    line2 = file.readline()
    file.close()

    line1 = line1.replace('cam0=', '')
    line2 = line2.replace('cam1=', '')

    K1 = np.matrix(line1)
    K2 = np.matrix(line2)

    return K1, K2


# calculate all possible combinations for R, t from E matrix
def E2Rt(E):
    U, S, Vt = np.linalg.svd(E)
    t1 = U[:, 2]
    t2 = -t1

    W = np.zeros((3,3))
    W[0, 1] = -1
    W[1, 0] = 1
    W[2, 2] = 1

    R1 = U @ W @ Vt
    R2 = U @ W.transpose() @ Vt

    return R1, R2, t1, t2


# calculate E matrix from F
# K1, K2: left and right intrinsic calibration matrix
def F2E(K1, K2, F):
    return K2.transpose() @ F @ K1


def printRtEpiLines(img1, img2, pts1, pts2, F, K1, K2):
    R1, R2, t1, t2 = E2Rt(F2E(K1, K2, F))
    print("\nPossible solutions for R, t:")
    print("R1:\n", R1, "\n\nR2:\n", R2, "\n\nt1:\n", t1, "\n\nt2:\n", t2, "\n\n")

    eimg = drawEpipolarLines(img1, img2, pts1, pts2, F)
    plt.imshow(cv.cvtColor(eimg,cv.COLOR_BGR2RGB))
    plt.show()
    return eimg
