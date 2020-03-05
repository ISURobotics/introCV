import cv2
import numpy as np

#we need this because the trackbars require it
def nothing(x):
    pass

#open the camera
cap = cv2.VideoCapture(0)

#populate the first frame
ret, frame = cap.read()

#make a named window so we can display trackbars
cv2.namedWindow('filtered image')

#make trackbars for filtering
#first value is label
#second is the window to display the trackbar on
#third is the default value
#fourth is the max value
#fifth is the function to call when something changes
cv2.createTrackbar('hue_lower', 'filtered image', 10, 255, nothing)
cv2.createTrackbar('hue_upper', 'filtered image', 255, 255, nothing)
cv2.createTrackbar('saturation', 'filtered image', 0, 255, nothing)
cv2.createTrackbar('value', 'filtered image', 0, 255, nothing)

#while able to read from the camera
while(ret):
    #the frame is mirrored originally, so mirror it again
    frame = cv2.flip(frame, 1)

    #convert bgr to hsv so it's easier to understand
    filtered = frame.copy()
    filtered = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)

    #create our range of values
    lower = np.array([cv2.getTrackbarPos('hue_lower', 'filtered image'), cv2.getTrackbarPos('saturation', 'filtered image'), cv2.getTrackbarPos('value', 'filtered image')], dtype = "uint8")
    upper = np.array([cv2.getTrackbarPos('hue_upper', 'filtered image'), 255, 255], dtype = "uint8")

    #filter and clean up
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.inRange(filtered, lower, upper)
    # mask = cv2.erode(mask, kernel, iterations = 2)
    # mask = cv2.dilate(mask, kernel, iterations = 2)
    mask = cv2.erode(mask, None, iterations = 2)
    mask = cv2.dilate(mask, None, iterations = 2)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)

    #display the correct values
    filtered = cv2.bitwise_and(frame, frame, mask = mask)

    #resize this because mine is too big
    dim = (int(frame.shape[1]*.5), int(frame.shape[0]*.5))
    frame = cv2.resize(frame, dim)
    filtered = cv2.resize(filtered, dim)

    #show what the camera sees
    cv2.imshow('camera image', frame)
    cv2.imshow('filtered image', filtered)

    #break if 'q' is pressed
    if(cv2.waitKey(1) == ord('q')):
        break

    #read the next frame
    ret, frame = cap.read()
