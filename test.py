import cv2
import numpy as np

# 非极大值抑制
def fastNonMaxSuppression(boxes, sc, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = sc
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the score of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick]

# 行人检测
# hog.load('myHogDector.bin') #因为在同一个文件中，不需要加载模型
hog = cv2.HOGDescriptor()
hog.load('myHogDector.bin')
image = cv2.imread("bpx/img/person_102.png")
# cv2.imshow("image", image)
# cv2.waitKey(0)
rects, scores = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)

# fastNonMaxSuppression第一个参数
for i in range(len(rects)):
    r = rects[i]
    rects[i][2] = r[0] + r[2]
    rects[i][3] = r[1] + r[3]

# fastNonMaxSuppression第二个参数
sc = [score[0] for score in scores]
sc = np.array(sc)

pick = []
print('rects_len', len(rects))
pick = fastNonMaxSuppression(rects, sc, overlapThresh=0.3)
print('pick_len = ', len(pick))

for (x, y, xx, yy) in pick:
    print(x, y, xx, yy)
    cv2.rectangle(image, (int(x), int(y)), (int(xx), int(yy)), (0, 0, 255), 2)
    cv2.imshow('a', image)
    cv2.waitKey(0)
