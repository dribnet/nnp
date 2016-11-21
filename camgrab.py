import cv2

#capture from camera at location 0
cap = cv2.VideoCapture(0)
# result1 = cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
# result2 = cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1024)
result1 = cap.set(cv2.CAP_PROP_FRAME_WIDTH,720)
result2 = cap.set(cv2.CAP_PROP_FRAME_HEIGHT,512)
print("Result is {}, {}".format(result1, result2))

while True:
    ret, img = cap.read()
    cv2.imshow("input", img)

    key = cv2.waitKey(10)
    if key == 27:
        break


cv2.destroyAllWindows() 
cv2.VideoCapture(0).release()
