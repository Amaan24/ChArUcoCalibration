# Amaan Vally
# Written using Python 3.8 and OpenCV 4.4.0
# Tested on Windows 10

import numpy
import cv2
import cv2.aruco as aruco
import os
import pickle

# Check if calibration data is availabe
if not os.path.exists('./calibration.pckl'):
    print("First calibrate your camera!")
    exit()
else:
    f = open('calibration.pckl', 'rb')
    (cameraMatrix, distCoeffs, _, _) = pickle.load(f)
    f.close()
    
    if cameraMatrix is None or distCoeffs is None:
        print("Something is missing from the calibration file. Please calbrate again.")
        exit()

# Constants for later
ARUCO_PARAMETERS = aruco.DetectorParameters_create()
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_50)
rows = 7
cols = 5

# This should correspond with GenerateCharucoBoard.py 
CHARUCO_BOARD = aruco.CharucoBoard_create(
        squaresX=rows,
        squaresY=cols,
        squareLength=0.051,
        markerLength=0.0255,
        dictionary=ARUCO_DICT)

rvec, tvec = None, None

#cam = cv2.VideoCapture(0) #Use this if youre using a connected camera instead of a video
cam = cv2.VideoCapture('EstimatePose.mp4')

while(cam.isOpened()):
    # Capture current frame
    ret, QueryImg = cam.read()
    if ret == True:
        # Convert image to grayscale
        dst = cv2.cvtColor(QueryImg, cv2.COLOR_BGR2GRAY)
    
        # Detect Aruco markers
        corners, ids, rejectedImgPoints = aruco.detectMarkers(dst, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
  
        # Refine detected markers
        corners, ids, rejectedImgPoints, recoveredIds = aruco.refineDetectedMarkers(
                image = dst,
                board = CHARUCO_BOARD,
                detectedCorners = corners,
                detectedIds = ids,
                rejectedCorners = rejectedImgPoints,
                cameraMatrix = cameraMatrix,
                distCoeffs = distCoeffs)   

        # Outline the markers
        QueryImg = aruco.drawDetectedMarkers(QueryImg, corners, borderColor=(0, 0, 255))

        # Only try to find CharucoBoard if markers were detected
        if ids is not None and len(ids) > 10:

            # Get charuco corners and ids from detected aruco markers
            response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                    markerCorners=corners,
                    markerIds=ids,
                    image=dst,
                    board=CHARUCO_BOARD)
    
            # Require more than 20 squares
            if response is not None and response > 20:
                pose, rvec, tvec = aruco.estimatePoseCharucoBoard(
                        charucoCorners=charuco_corners, 
                        charucoIds=charuco_ids, 
                        board=CHARUCO_BOARD, 
                        cameraMatrix=cameraMatrix, 
                        distCoeffs=distCoeffs,
                        rvec=rvec,
                        tvec=tvec)

                if pose:
                    # Draw axis on screen
                    QueryImg = aruco.drawAxis(QueryImg, cameraMatrix, distCoeffs, rvec, tvec, 0.051)
                    print("RVEC")
                    print(rvec)
                    print("TVEC")
                    print(tvec)
                    
            
        # Display
        cv2.imshow('QueryImage', QueryImg)

    # Exit at the end of the video when q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
