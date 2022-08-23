import cv2

img = "C:\\Users\\HP\\Desktop\\Research\\images\\img1.jpg"
a = cv2.imread(img)
print(a.shape)
for x in range(a.shape[0]):
    for y in range(a.shape[1]):
       b = a[x][y]
       a[x][y] = (0, b[1], b[2])
       print(1)
cv2.imwrite("c.jpg", a)
