import cv2

p = "D:\\securityenhancement\\static\\images\\img1.jpg"
img = cv2.imread(p)
a = img[0:40, 0:40]
i = cv2.imwrite("a.jpg", a)