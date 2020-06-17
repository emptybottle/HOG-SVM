import cv2
import numpy as np

hog = cv2.HOGDescriptor()
hog.load('myHogDectorP.bin')
cap = cv2.VideoCapture(0)
while True:
    img =cv2.imread('crop_000010.png')
    # ok, img = cap.read()
    rects, wei = hog.detectMultiScale(img,winStride=(4, 4), padding=(8, 8), scale=1.05)
    print(rects)
    for (x, y, w, h) in rects:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imshow('test', img)
    if cv2.waitKey(1) & 0xff == 27:  # esc键
        break
cv2.destroyAllWindows()
