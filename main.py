import cv2

filepath = "anime.gif"
cap = cv2.VideoCapture(filepath)

img_sum = None
img_src = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_src = frame
    if img_sum is None:
        img_sum = img_src
    else:
        img_sum = cv2.addWeighted(img_sum, 0.6, img_src, 0.4, 0)

    cv2.imshow("Frame", img_sum)
    key = cv2.waitKey(30)
    if key == 27:
        break

cv2.imwrite('hoge.jpg', img_sum)
cap.release()
cv2.destroyAllWindows()