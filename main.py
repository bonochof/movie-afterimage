import cv2
import numpy as np

filepath = "anime.gif"
cap = cv2.VideoCapture(filepath)
fgbg = cv2.createBackgroundSubtractorMOG2()

first = None
img_sum = None
img_src = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    fgmask = fgbg.apply(frame)
    fgmask = cv2.medianBlur(fgmask, 5)
    img_src = frame
    if img_sum is None:
        img_sum = img_src
    else:
        img_sum_rgba = cv2.cvtColor(img_sum, cv2.COLOR_RGB2RGBA)
        img_src_and = cv2.bitwise_and(frame, frame, mask=fgmask)
        img_src_rgba = cv2.cvtColor(img_src_and, cv2.COLOR_RGB2RGBA)
        img_src_rgba[..., 3] = np.where(np.all(img_src_and==0, axis=-1), 0, 255)
        img_sum_add = img_sum_rgba + img_src_rgba
        #img_sum = cv2.bitwise_or(img_sum_rgba, img_src_rgba)
        img_sum = cv2.addWeighted(img_sum_add, 1.0, img_src_rgba, 1.0, 0)
        
        if first is None:
            first = img_sum

    cv2.imshow("Frame", img_sum)
    key = cv2.waitKey(30)
    if key == 27:
        break

img_sum = cv2.addWeighted(img_sum, 0.6, first, 0.4, 0)

cv2.imwrite('hoge.jpg', img_sum)
cap.release()
cv2.destroyAllWindows()