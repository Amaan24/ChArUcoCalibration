# System information:
# - Linux Mint 18.1 Cinnamon 64-bit
# - Python 2.7 with OpenCV 3.2.0

import numpy 
import cv2
from cv2 import aruco
import pickle
import glob


# ChAruco board variables
CHARUCOBOARD_ROWCOUNT = 7
CHARUCOBOARD_COLCOUNT = 5 
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_1000)

# Create constants to be passed into OpenCV and Aruco methods
CHARUCO_BOARD = aruco.CharucoBoard_create(
        squaresX=CHARUCOBOARD_COLCOUNT,
        squaresY=CHARUCOBOARD_ROWCOUNT,
        squareLength=0.051,
        markerLength=0.0255,
        dictionary=ARUCO_DICT)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# Create the arrays and variables we'll use to store info like corners and IDs from images processed
corners_all = [] # Corners discovered in all images processed
ids_all = [] # Aruco ids corresponding to corners discovered
image_size = None # Determined at runtime

objectPoints = []
imagePointsLeft = []
imagePointsRight = []

objp = numpy.zeros((4*6, 3), numpy.float32)
objp[:, :2] = 0.051*numpy.mgrid[0:4, 0:6].T.reshape(-1, 2)

# This requires a set of images or a video taken with the camera you want to calibrate
# I'm using a set of images taken with the camera with the naming convention:
# 'camera-pic-of-charucoboard-<NUMBER>.jpg'
# All images used should be the same size, which if taken with the same camera shouldn't be a problem
imagesLeft = glob.glob('C:/Users//User//Desktop//Mechatronics 2020//Second Semester//EEE4022S//Code//Calibration Images//Main//LeftCamera//Stereo//Use//*.jpg')
imagesRight = glob.glob('C:/Users//User//Desktop//Mechatronics 2020//Second Semester//EEE4022S//Code//Calibration Images//Main//RightCamera//Stereo//Use//*.jpg')

imagesLeft.sort()
imagesRight.sort()


# Loop through images glob'ed
for i, fname in enumerate(imagesLeft):
    # Open the images
    imgL = cv2.imread(imagesLeft[i])
    imgR = cv2.imread(imagesRight[i])
    
    # Grayscale the image
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # Find aruco markers in the query image
    cornersL, idsL, _ = aruco.detectMarkers(image=grayL,dictionary=ARUCO_DICT)
    cornersR, idsR, _ = aruco.detectMarkers(image=grayR,dictionary=ARUCO_DICT)

    # Outline the aruco markers found in our query image
    imgL = aruco.drawDetectedMarkers(image=imgL, corners=cornersL)
    imgR = aruco.drawDetectedMarkers(image=imgR, corners=cornersR)

    # Get charuco corners and ids from detected aruco markers
    responseL, charuco_cornersL, charuco_idsL = aruco.interpolateCornersCharuco(markerCorners=cornersL,markerIds=idsL,image=grayL,board=CHARUCO_BOARD)
    responseR, charuco_cornersR, charuco_idsR = aruco.interpolateCornersCharuco(markerCorners=cornersR,markerIds=idsR,image=grayR,board=CHARUCO_BOARD)

    # If a Charuco board was found, let's collect image/corner points
    # Requiring all squares
    if responseL == 24 and responseR == 24:
        # Add these corners and ids to our calibration arrays
        objectPoints.append(objp)
        
        imgL = aruco.drawDetectedCornersCharuco(
                image=imgL,
                charucoCorners=charuco_cornersL,
                charucoIds=charuco_idsL)
        imgR = aruco.drawDetectedCornersCharuco(
                image=imgR,
                charucoCorners=charuco_cornersR,
                charucoIds=charuco_idsR)
        #cv2.imshow('Charuco board Left', imgL)
        #cv2.waitKey(0)
        #cv2.imshow('Charuco board Right', imgR)
        #cv2.waitKey(0)
        
        
    

        rt = cv2.cornerSubPix(grayL, charuco_cornersL, (11, 11),(-1, -1), criteria)
        imagePointsLeft.append(charuco_cornersL)

        rt = cv2.cornerSubPix(grayR, charuco_cornersR, (11, 11),(-1, -1), criteria)
        imagePointsRight.append(charuco_cornersR)
        
        # If our image size is unknown, set it now
        if not image_size:
            image_size = grayL.shape[::-1]

        print("Done with " + fname)
    else:
        print("Not able to detect a charuco board in left and right image pair")

# Destroy any open CV windows
cv2.destroyAllWindows()

# Make sure at least one image was found
if len(imagesLeft) < 1:
    # Calibration failed because there were no images, warn the user
    print("Calibration was unsuccessful. No images of charucoboards were found. Add images of charucoboards and use or alter the naming conventions used in this file.")
    # Exit for failure
    exit()

# Make sure we were able to calibrate on at least one charucoboard by checking
# if we ever determined the image size
if not image_size:
    # Calibration failed because we didn't see any charucoboards of the PatternSize used
    print("Calibration was unsuccessful. We couldn't detect charucoboards in any of the images supplied. Try changing the patternSize passed into Charucoboard_create(), or try different pictures of charucoboards.")
    # Exit for failure
    exit()

# Now that we've seen all of our images, perform the camera calibration
# based on the set of points we've discovered

stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

f = open('calibrationLeftNew.pckl', 'rb')
(cameraMatrixLeft, distCoeffsLeft, _, _) = pickle.load(f)
f.close()
f = open('calibrationRightNew.pckl', 'rb')
(cameraMatrixRight, distCoeffsRight, _, _) = pickle.load(f)
f.close()

ret, cameraMatrixLeft, distCoeffsLeft, cameraMatrixRight, distCoeffsRight, R, T, E, F = cv2.stereoCalibrate(
            objectPoints, imagePointsLeft,
            imagePointsRight, cameraMatrixLeft, distCoeffsLeft, cameraMatrixRight,
            distCoeffsRight, image_size,
            criteria=stereocalib_criteria, flags= (cv2.CALIB_FIX_INTRINSIC+cv2.CALIB_USE_INTRINSIC_GUESS+
                                                   cv2.CALIB_FIX_K3+cv2.CALIB_FIX_K4+cv2.CALIB_FIX_K5)+
                                                    cv2.CALIB_ZERO_TANGENT_DIST+ cv2.CALIB_RATIONAL_MODEL +
                                                    cv2.CALIB_FIX_PRINCIPAL_POINT
            )
    
# Print matrix and distortion coefficient to the console
print(ret)
#print(cameraMatrixLeft)
#print(cameraMatrixRight)
#print(distCoeffsLeft)
#print(distCoeffsRight)
print(R)
print(T)
#print(E)
#print(F)

#Store Returned values in pickle file for later use
#f = open('StereoCalibrationResults.pckl', 'wb')
#pickle.dump((cameraMatrixLeft, cameraMatrixRight, distCoeffsLeft, distCoeffsRight, R, T, E, F), f)
#f.close()

#Notify that stereocalibration is done
print('StereoCalibration completed successfully!')
