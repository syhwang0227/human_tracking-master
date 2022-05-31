import cv2
import numpy as np

## Read and copy
img = cv2.imread("Mediapipe_screenshot_10.09.2021.png")
canvas = img.copy()

## set and crop the ROI
x,y,w,h = bbox = (180, 100, 50, 100)
cv2.rectangle(canvas, (x,y), (x+w,y+h), (0,0,255), 2)
# cv2.rectangle(canvas, (x,y+200), (x+w,(y+200)+h), (0,0,255), 2)

croped = img[y:y+h, x:x+w]
# croped2 = img[(y-20):(y+h)+20, (x-20):(x+w)+20]
cv2.imshow("croped", croped)
# cv2.imshow("croped2", croped2)


## get the center and the radius
cx = x+w//2
print("cx", cx)
cy = y+h//2
print("cy", cy)
cr = max(w, h)//2
print("max:", max(w, h))
print("cr", cr)

## set offset, repeat enlarger ROI
dr = 10
for i in range(0,4):
    r = cr+dr
    print("r", r)
    cv2.rectangle(canvas, (cx-r, cy-r), (cx+r, cy+r), (0,255,0), 1)
    croped = img[cy-r:cy+r, cx-r:cx+r]
    # cv2.imshow("croped{}".format(i), croped)

## display
cv2.imshow("source", canvas)
cv2.waitKey()
cv2.destroyAllWindows()