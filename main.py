import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# FOR IMAGE DEMONSTRATION
image = cv.imread('laneimage.jpg')

# TO SHOW NORMAL IMAGE
lane_image = np.copy(image)
cv.imshow('road normal', lane_image)
cv.waitKey(0)

# CONVERT IMAGE TO GRAYSCALE
gray = cv.cvtColor(lane_image, cv.COLOR_BGR2GRAY)
cv.imshow('road grayscale', gray)
cv.waitKey(0)

# SMOOTHEN OUT GRAY IMAGE AND HIGHLIGHT EDGES
smoothed_image = cv.GaussianBlur(gray, (5,5), 0)
canny_image = cv.Canny(smoothed_image, 50, 150)
cv.imshow('road with edges', canny_image)
cv.waitKey(0)

# FUNCTION TO CREATE MASK FOR REGION OF INTEREST
def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[
        (270, 710), (900, 710), (630, 515)]
    ])
    mask = np.zeros_like(image)
    cv.fillPoly(mask, polygons, 255)
    masked_image = cv.bitwise_and(image, mask)
    return masked_image

cropped_image = region_of_interest(canny_image)

# HOUGH TRANSFORM TO DETECT LINES
lines = cv.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=50)

# FUNCTION DISPLAYS LINES ON MASK
def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
    return line_image

# DISPLAYS DETECTED LINES ON MASK
line_image = display_lines(lane_image, lines)
cv.imshow('lines of detection', line_image)
cv.waitKey(0)

# OVERLAPS LINES WITH LANE IMAGE
combo_image = cv.addWeighted(lane_image, 0.8, line_image, 1, 1)
cv.imshow('road with lines', combo_image)
cv.waitKey(0)

# FOR VIDEO, LINES ARE DETECTED IN EACH FRAME
cap = cv.VideoCapture('LaneDetectionTestVideo.mp4')
while (cap.isOpened()):

    _, frame = cap.read()

    print(frame)

    if (frame is not None):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    else:
        print('\n')
        print('VIDEO FINISHED')
        exit(0)
    smoothed_image = cv.GaussianBlur(gray, (5, 5), 0)
    canny_image = cv.Canny(smoothed_image, 50, 150)
    cropped_image = region_of_interest(canny_image)
    lines = cv.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=200)
    line_image = display_lines(frame, lines)
    combo_image = cv.addWeighted(frame, 0.8, line_image, 1, 1)
    cv.imshow("final result", combo_image)

    if cv.waitKey(1) == ord('q'):
        exit(0)

cap.release()
cap.destroyAllWindows()


