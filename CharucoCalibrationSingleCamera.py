# Amaan Vally
# Written using Python 3.8 and OpenCV 4.4.0
# Tested on Windows 10

import pickle #To store parameters to a file
import glob #To manage images
import numpy
import cv2
from cv2 import aruco


# ChArUco Information
rows = 7
cols = 5
dictionary = aruco.Dictionary_get(aruco.DICT_5X5_1000)

# This should correspond with GenerateCharucoBoard.py
CHARUCO_BOARD = aruco.CharucoBoard_create(
        squaresX=rows,
        squaresY=cols,
        squareLength=0.051,
        markerLength=0.0255,
        dictionary=ARUCO_DICT)

# Create the arrays and variables we'll need later
ids_all = [] # All Aruco ids from all images
corners_all = [] # All corners from all images
image_size = None # Determined from provided images


# All images should be the same size and in the same folder
images = glob.glob('C:/Users//User//Desktop//Mechatronics 2020//Second Semester//EEE4022S//Code//Calibration Images//Main//RightCamera//Single//*.jpg')

# Loop through images
for iname in images:
    # Open the image
    img = cv2.imread(iname)
    
    # Convert to grayscale
    dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find ArUco markers
    corners, ids, _ = aruco.detectMarkers(
            image=dst,
            dictionary=ARUCO_DICT)

    # Outline ArUco markers 
    img = aruco.drawDetectedMarkers(
            image=img, 
            corners=corners
            )

    # Interpolate positions of corners
    retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=dst,
            board=CHARUCO_BOARD)

    # If at least 18 corners are interpolated 
    if retval > 18:
        # Append to array for later
        corners_all.append(charuco_corners)
        ids_all.append(charuco_ids)
        
        # Draw the Charuco board
        img = aruco.drawDetectedCornersCharuco(
                image=img,
                charucoCorners=charuco_corners,
                charucoIds=charuco_ids)
       
        # Set image size
        if not image_size:
            image_size = dst.shape[::-1]
        
        print("Done with image: {}".format(iname))
    
        # Adjsut size so can be displayed on screen
        proportion = max(img.shape) / 750
        img = cv2.resize(img, (int(img.shape[1]/proportion), int(img.shape[0]/proportion)))

        # Display images, waiting for key press after each one
        cv2.imshow('Charuco board', img)
        cv2.waitKey(0)
    else:
        print("Not able to detect a complete charuco board in image: {}".format(iname))

# Destroy any open CV windows
cv2.destroyAllWindows()

# Make sure at least one image was found
if len(images) < 1:
    print("Calibration was unsuccessful. No images of charucoboards were found.")
    exit()

if not image_size:
    print("Calibration was unsuccessful. No ChArUco board detected.")
    exit()

# Perform the calibration
err, cameraMatrix, distCoeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
        charucoCorners=corners_all,
        charucoIds=ids_all,
        board=CHARUCO_BOARD,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None)
    
# Print to console
print(cameraMatrix)
print(distCoeffs)
print(rvecs)
print(tvecs)
print(err)

# Save to pickle file for later use
f = open('calibration.pckl', 'wb')
pickle.dump((cameraMatrix, distCoeffs, rvecs, tvecs), f)
f.close()

# Print to console
print('Calibration successful')
