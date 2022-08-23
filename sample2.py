import cv2
img = "C:\\Users\\HP\\Desktop\\Research\\images\\img1.jpg"
a = cv2.imread(img,0)
x,y = a.shape
print(x,y)
print(a[0,0])
