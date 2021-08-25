from keras.models import load_model

model = load_model('best_model.h5')

letters ={ 0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f', 6:'g', 7:'h', 8:'i', 9:'j', 10:'k', 11:'l', 
          12:'m', 13:'n', 14:'o', 15:'p', 16:'q', 17:'r', 18:'s', 19:'t', 20:'u', 21:'v', 22:'w', 
          23:'x', 24:'y', 25:'z', 26:''}

# defining blue color in hsv format
# pip install numpy
import numpy as np

blueLower = np.array([100,60,60])
blueUpper = np.array([140,255,255])

kernel = np.ones((5,5), np.uint8)

# define blackboard
blackboard = np.zeros((480,640, 3), dtype=np.uint8)
alphabet = np.zeros((200,200,3), dtype=np.uint8)

# deques (Double ended queue) is used to store alphabet drawn on screen
from collections import deque
points = deque(maxlen = 512)



### open the camera and recognize alphabet

import cv2 #pip install opencv-python
cap = cv2.VideoCapture(0)

while True:
    ret, frame=cap.read()
    frame = cv2.flip(frame, 1)
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detecting which pixel value falls under blue color boundaries
    blue = cv2.inRange(hsv, blueLower, blueUpper)
    
    #erosion
    blue = cv2.erode(blue, kernel)
    #opening
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, kernel)
    #dilation
    blue = cv2.dilate(blue, kernel)
    
    # find countours in the image
    cnts , _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
     
    center = None
    
    # if any countours were found
    if len(cnts) > 0:
        cnt = sorted(cnts, key = cv2.contourArea, reverse=True)[0]
        ((x,y), radius) = cv2.minEnclosingCircle(cnt)
        cv2.circle(frame, (int(x), int(y),), int(radius), (125,344,278), 2)
        
        M = cv2.moments(cnt)
        center = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))
    
        points.appendleft(center)
        
    elif len(cnts) == 0:
        if len(points) != 0:
            blackboard_gray = cv2.cvtColor(blackboard, cv2.COLOR_BGR2GRAY)
            blur = cv2.medianBlur(blackboard_gray, 15)
            blur = cv2.GaussianBlur(blur, (5,5), 0)
            thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
            #cv2.imshow("Thresh", thresh)
            
    
    cv2.imshow("Alphabet Recognition System", frame)
    
    if cv2.waitKey(1)==13: #if I press enter
        break
cap.release()
cv2.destroyAllWindows()