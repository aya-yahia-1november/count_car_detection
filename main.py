
import cv2
from time import sleep
import numpy as np
def calculate_center(x,y,w,h):
    x1=int(w/2)
    y1=int(h/2)
    cx=x+x1
    cy=y+y1
    return cx,cy

y1=200
ww=30
hh=30
detect=[]
out_line=3
delay=30
count=0
cap =cv2.VideoCapture("cars.mp4")
back_ground_sub=cv2.bgsegm.createBackgroundSubtractorMOG()
while True:
    _,frame=cap.read()
    temp = float(1 / delay)
    sleep(temp)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(gray,(3,3),5)
    img_sub = back_ground_sub.apply(blur)
    dilated=cv2.dilate(img_sub,np.ones((5,5)))
    contours,_=cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.line(frame,(5,y1),(1042,y1),(0,255,50),3)
    for (i,coordinates) in enumerate(contours):
        x,y,w,h=cv2.boundingRect(coordinates)
        detection_area=(w>=ww)and(h>=hh)
        if not detection_area:
            continue
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        center=calculate_center(x,y,w,h)
        detect.append(center)
        cv2.circle(frame,center,3,(13,250,0),-1)
        #count the cars
        for (x,y) in detect:
            if (y < (y1 + out_line)) and (y > (y1- out_line)):
                count+=1
                cv2.line(frame,(10,y1),(1042,y1), (123, 50, 10), 3)
                detect.remove((x,y))
                print("number of cars is"+str(count))

    cv2.putText(frame,"CARS COUNT IS "+str(count),(0,30),cv2.FONT_HERSHEY_SIMPLEX,1,(100,255,50),2)
    cv2.imshow("orginl video",frame)
    cv2.imshow("dilatrd",dilated)
    cv2.imshow("img_sub",img_sub)
    if cv2.waitKey(1) == 27:
        break
cv2.destroyWindow()
cap.release()
