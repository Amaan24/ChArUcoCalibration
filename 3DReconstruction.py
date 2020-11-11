import cv2
import numpy 
import pickle
import matplotlib.pyplot as plt
import math

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#Load Output Matrices from StereoCalibrate
f = open('StereoCalibrationResults.pckl', 'rb')
(cameraMatrixLeft, cameraMatrixRight, distCoeffsLeft, distCoeffsRight, R, T, E, F) = pickle.load(f)
f.close()
print(T)
print(R)
#Load Pair of images for triangulation
imgLeft = cv2.imread('C:/Users//User//Desktop//Mechatronics 2020//Second Semester//EEE4022S//Code//Calibration Images//Main//LeftCamera//triangulationLeft.jpg')
imgRight = cv2.imread('C:/Users//User//Desktop//Mechatronics 2020//Second Semester//EEE4022S//Code//Calibration Images//Main//RightCamera//triangulationRight.jpg')

#Grayscale
imgLeftGray = cv2.cvtColor(imgLeft, cv2.COLOR_BGR2GRAY)
imgRightGray = cv2.cvtColor(imgRight, cv2.COLOR_BGR2GRAY)

image_size = imgLeftGray.shape[::-1]

#Rectification
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(cameraMatrixLeft, distCoeffsLeft, cameraMatrixRight, distCoeffsRight, image_size, R, T, alpha=1)

leftMapX, leftMapY = cv2.initUndistortRectifyMap(cameraMatrixLeft, distCoeffsLeft, R1, P1, image_size, cv2.CV_32FC1)
rightMapX, rightMapY =cv2.initUndistortRectifyMap(cameraMatrixRight, distCoeffsRight, R2, P2, image_size, cv2.CV_32FC1)

dstLeft = cv2.remap(imgLeftGray, leftMapX, leftMapY, cv2.INTER_LINEAR)
dstRight = cv2.remap(imgRightGray, rightMapX, rightMapY, cv2.INTER_LINEAR)


#Draw rectangle around ROIs
x1,y1,w1,h1 = validPixROI1
x2,y2,w2,h2 = validPixROI2

dstLeft = cv2.rectangle(dstLeft, (x1,y1), (w1,h1), (255,0,0), 2)
dstRight = cv2.rectangle(dstRight, (x2,y2), (w2,h2), (255,0,0), 2)

#dstLeft = dstLeft[y1:y1+h1, x1:x1+w1]
#dstRight = dstRight[y1:y1+h1, x1:x1+w1]


#Combining images side by side
combined = cv2.hconcat([dstLeft, dstRight])

# Reproportion the image, maxing width or height at 1500
proportion = max(combined.shape) / 1500.0
combined = cv2.resize(combined, (int(combined.shape[1]/proportion), int(combined.shape[0]/proportion)))


#Draw Horizontal lines. This is equivalent to drawing epipolar lines on non-rectified images
for l in range(30):
     yVal = int(combined.shape[0]/30)*l
     cv2.line(combined, pt1=(0,yVal), pt2=(2000,yVal), color=(255,0,0), thickness=1)

combined = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)

# Pause to display next to each other and wait
cv2.imshow('Combined', combined)
cv2.waitKey(0)

#Saving images to png files
cv2.imwrite('rectLeft.png', dstLeft)
cv2.imwrite('rectRight.png', dstRight)
cv2.imwrite('rectBothWithEpilines.png', combined)

#Create a depth map
stBM = cv2.StereoSGBM_create(numDisparities=240, blockSize=21)
disparity = stBM.compute(dstLeft,dstRight)
plt.imshow(disparity, 'gray')
plt.show()


#Find POIs
ret_l, cornersLeft = cv2.findChessboardCorners(dstLeft, (8, 6), None)
ret_r, cornersRight = cv2.findChessboardCorners(dstRight, (8, 6), None)

if ret_l and ret_r:
     cornersLeft = cv2.cornerSubPix(dstLeft,cornersLeft,(11,11),(-1,-1),criteria)
     cornersRight = cv2.cornerSubPix(dstRight,cornersRight,(11,11),(-1,-1),criteria)

# If found, add object points, image points (after refining them)
if ret_l and ret_r:

    # Draw and display the corner
    img = cv2.drawChessboardCorners(dstLeft, (8,6), cornersLeft,ret_l)
    cv2.imshow('img',img)
    cv2.waitKey(0)
    # Draw and display the corners
    img = cv2.drawChessboardCorners(dstRight, (8,6), cornersRight,ret_r)
    cv2.imshow('img',img)
    cv2.waitKey(0)
    
cv2.destroyAllWindows()

#Triangulate
imgPtsLeft = numpy.zeros((48,2)) 
imgPtsRight = numpy.zeros((48,2))
wrldPts = numpy.zeros((48,4,1))
distance = numpy.zeros((48,1))

for i in range(len(cornersLeft)):
    imgPtsLeft[i] = numpy.array([cornersLeft[i]], dtype=numpy.float)
    imgPtsRight[i] = numpy.array([cornersRight[i]], dtype=numpy.float)

for k in range(len(imgPtsLeft)):
    wrldPts[k] = cv2.triangulatePoints(P1,P2,imgPtsLeft[k],imgPtsRight[k])
    wrldPts[k] /= wrldPts[k][3]

for h in range(len(wrldPts)-1):
     distance[h] = numpy.linalg.norm(wrldPts[h] - wrldPts[h+1])  

for g in range(len(distance)):
     print(distance[g])

distanceV = numpy.zeros((40,1))

for f in range(39):
     distanceV[f] = numpy.linalg.norm(wrldPts[f] - wrldPts[f+8])
for a in range(len(distanceV)):
     print(distanceV[a])

distance = distance[(distance < 0.1) & (distance > 0.02)]        
distanceV = distanceV[(distanceV < 0.1) & (distanceV > 0.02)]

print("Average Horizontal:")
print(numpy.average(distance))
     
print("Average Vertical:")
print(numpy.average(distanceV))

errH = numpy.sqrt(((distance - 0.04) ** 2).mean())
errV = numpy.sqrt(((distanceV - 0.04) ** 2).mean())

print(errH)
print(errV)

xline = numpy.zeros((48,1))
yline = numpy.zeros((48,1))
zline = numpy.zeros((48,1))

xline2 = numpy.zeros((4,1))
yline2 = numpy.zeros((4,1))
zline2 = numpy.zeros((4,1))

for y in range(len(wrldPts)):
     xline[y] = wrldPts[y][0]
     yline[y] = wrldPts[y][1]
     zline[y] = wrldPts[y][2]

xline2[0] = xline[0]
yline2[0] = yline[0]
zline2[0] = zline[0]

xline2[1] = xline[7]
yline2[1] = yline[7]
zline2[1] = zline[7]

xline2[2] = xline[40]
yline2[2] = yline[40]
zline2[2] = zline[40]

xline2[3] = xline[47]
yline2[3] = yline[47]
zline2[3] = zline[47]


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(xline.flatten(), yline.flatten(), zline.flatten())
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()






