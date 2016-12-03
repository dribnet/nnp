import cv2

#capture from camera at location 0
cam= cv2.VideoCapture(0)

cam_width = 400
cam_height = 300
result1 = cam.set(cv2.CAP_PROP_FRAME_WIDTH,cam_width)
result2 = cam.set(cv2.CAP_PROP_FRAME_HEIGHT,cam_height)
result3 = cam.set(cv2.CAP_PROP_FPS,1)
print("Result is {}, {}, {}".format(result1, result2, result3))

while True:
    ret, img = cam.read()
    cv2.imshow("input", img)

    key = cv2.waitKey(10)
    if key == 27:
        break


cv2.destroyAllWindows() 
cv2.VideoCapture(0).release()
