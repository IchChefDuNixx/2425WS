# exercise for fundamental matrix/8-point algorithm
from reco3dthro import *
# commented out, as we will discuss this in one of the next sessions
# from triangulation import *



# Normalize points pts using 3x3 transformation matrix T
def normalizePoints(pts, T):
    normpts = pts.astype(np.float64)
    for i in range(len(pts)):
        p = T @ np.matrix([[pts[i][0]],[pts[i][1]], [1]])
        normpts[i][0] = p[0,0]
        normpts[i][1] = p[1,0]

    return normpts


# compute 3x3 matrix for Hartley normalization of points
# translate to center
# scale so that mean squared distance between the origin and the points is 2 pixels
def computeNormalizationT(pts):
    T = np.identity(3)

    # todo

    return T


def computeNormalizedF(pts1, pts2):
    # just to have a matrix to return in the template
    F = np.zeros((3, 3))
    F[0, 0] = 1
    F[1, 1] = 1

    # todo
    # determine the two normalization matrices T1 and T2

    # use it to normalize the input points with "normalizePoints()"
    # and store the normalized points in normpts1, normpts2

    # compute F from normalized points

    # transform the resulting F using T1 and T2 to obtain the real F

    return F


def computeF(pts1, pts2):
    assert pts1.shape == pts2.shape, "pts1, pts2 must have the same shape"

    # just to have a matrix to return in the template
    F = np.zeros((3, 3))
    F[0, 0] = 1
    F[1, 1] = 1

    # todo
    # set up coefficient matrix A for 8-point algorithm

    # SVD

    # built 3x3 fundamental matrix from solution vector

    # enforce rank(F) = 2

    return F


def main():
    datafolder = 'Motorcycle-perfect/'

    # read images
    img1 = cv.imread(datafolder + 'im0.png', cv.IMREAD_GRAYSCALE)  # left image
    img2 = cv.imread(datafolder + 'im1.png', cv.IMREAD_GRAYSCALE)  # right image

    # read calibration data
    K1, K2 = readCalibFile(datafolder + 'calib.txt')

    # find left-right point matches
    pts1, pts2 = findpointmatches(img1,img2)
    print('Found {} matches'.format(len(pts1)))

    # compute fundamental matrices
    print("\nCompute F without normalization")
    F = computeF(pts1, pts2)
    eimg1 = printRtEpiLines(img1, img2, pts1, pts2, F, K1, K2)
    cv.imwrite(datafolder + 'F-nonorm.jpg', eimg1)

    # commented out, as we will discuss this in one of the next sessions
    # save3DReconstruction(K1, K2, F, pts1, pts2, datafolder + 'reconstructionnonorm.mat')

    # remove comment in the following 4 lines to call the normalized version
    # print("\nCompute F with normalization")
    # F = computeNormalizedF(pts1, pts2)
    # eimg2 = printRtEpiLines(img1, img2, pts1, pts2, F, K1, K2)
    # cv.imwrite(datafolder + 'F-norm.jpg', eimg2)

    # commented out, as we will discuss this in one of the next sessions
    # save3DReconstruction(K1, K2, F, pts1, pts2, datafolder + 'reconstructionnorm.mat')



if __name__ == '__main__':
    main()
