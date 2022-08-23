import cv2
img = "C:\\Users\\HP\\Desktop\\Research\\images\\img1.jpg"
a = cv2.imread(img,0)
cv2.imwrite("a.jpg",a)
cv2.imshow("title",a)
cv2.waitKey(0)