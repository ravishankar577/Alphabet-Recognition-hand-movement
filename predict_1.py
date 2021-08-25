import cv2
import numpy as np
from keras.models import load_model
from collections import deque
model = load_model('best_model.h5')
#pip install opencv-python
cap = cv2.VideoCapture(0)
prediction = 26

letters ={ 0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f', 6:'g', 7:'h', 8:'i', 9:'j', 10:'k', 11:'l', 
          12:'m', 13:'n', 14:'o', 15:'p', 16:'q', 17:'r', 18:'s', 19:'t', 20:'u', 21:'v', 22:'w', 
          23:'x', 24:'y', 25:'z', 26:''}

blueLower = np.array([100,60,60])
blueUpper = np.array([140,255,255])

kernel = np.ones((5,5), np.uint8)

from collections import deque
points = deque(maxlen = 512)
# define blackboard
blackboard = np.zeros((480,640, 3), dtype=np.uint8)
alphabet = np.zeros((200,200,3), dtype=np.uint8)
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
            
            blackboard_cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
            
            if len(blackboard_cnts)>=1:
                cnt = sorted(blackboard_cnts, key=cv2.contourArea, reverse=True)[0]
                
                if cv2.contourArea(cnt)>1000:
                    x,y,w,h = cv2.boundingRect(cnt)
                    alphabet = blackboard_gray[y-10:y+h+10,x-10:x+w+10]
                    try:
                        img = cv2.resize(alphabet, (28,28))
                    except cv2.error as e:
                        continue
                    
                    img = np.array(img)
                    img = img.astype('float32')/255
                    
                    prediction = model.predict(img.reshape(1,28,28))[0]
                    prediction = np.argmax(prediction)
                    
            # Empty the point deque and also blackboard
            points = deque(maxlen=512)
            blackboard = np.zeros((480,640, 3), dtype=np.uint8)
        
    # connect the detected points with line
    for i in range(1, len(points)):
        if points[i-1] is None or points[i] is None:
            continue
        cv2.line(frame, points[i-1], points[i], (0,0,0), 2)
        cv2.line(blackboard, points[i-1], points[i], (255,255,255), 8)
        
    
    cv2.putText(frame, "Prediction: " + str(letters[int(prediction)]), (20,400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    
    
    cv2.imshow("Alphabet Recognition System", frame)
    
    if cv2.waitKey(1)==13: #if I press enter
        break
cap.release()
cv2.destroyAllWindows()