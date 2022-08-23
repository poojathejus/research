import cv2
img = "C:\\Users\\HP\\Desktop\\Research\\images\\img1.jpg"
a = cv2.imread(img)
cv2.imwrite("b.jpg",a)
c = cv2.cvtColor(a,cv2.COLOR_RGB2YCrCb)
cv2.imshow("title",c)
cv2.waitKey(0)