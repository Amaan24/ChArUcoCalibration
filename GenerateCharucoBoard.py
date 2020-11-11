# Amaan Vally
# Written using Python 3.8 and OpenCV 4.4.0
# Tested on Windows 10
import cv2
import cv2.aruco as aruco

# Create ChArUco board
# Adjust squaresX, squaresY, squareLength and markerLength as desired
gridboard = aruco.CharucoBoard_create(
        squaresX=5, 
        squaresY=7, 
        squareLength=0.045, 
        markerLength=0.0275, 
        dictionary=aruco.Dictionary_get(aruco.DICT_5X5_1000))

# Create an image from the gridboard and store to "ChArUco Board.jpg"
# 96 DPI jpg, so image size max 1123x1587 for A3
img = gridboard.draw(outSize=(1123, 1587))
cv2.imwrite("ChArUco Board.jpg", img)

# Show the ChArUco board on screen
cv2.imshow('ChArUco Board', img)

# Exit when a key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()
