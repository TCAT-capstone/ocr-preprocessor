import numpy as np
import cv2

src = cv2.imread('./data/data4.jpg', 1)

# grayscale
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0)

# canny
canned = cv2.Canny(gray, 150, 300)

# dilate to close holes in lines
kernel = np.ones((10,1),np.uint8) # 가로 1 세로 10
mask = cv2.dilate(canned, kernel, iterations = 20)

# contours 찾기
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 가장 큰 contours 찾기
biggest_cntr = None
biggest_area = 0
for contour in contours:
    area = cv2.contourArea(contour)
    if area > biggest_area:
        biggest_area = area
        biggest_cntr = contour

# 외곽 box
rect = cv2.minAreaRect(biggest_cntr)
box = cv2.boxPoints(rect)
box = np.int0(box)

# 외곽 box 그리기
src_box = src.copy()
cv2.drawContours(src_box, [box], 0, (0, 255, 0), 3)

# angle 계산
angle = rect[-1]
if angle > 45:
    angle = -(90 - angle)

# 기울기 조정
rotated = src.copy()
(h, w) = rotated.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(rotated, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# 회전된 박스 좌표 찾기
ones = np.ones(shape=(len(box), 1))
points_ones = np.hstack([box, ones])
transformed_box = M.dot(points_ones.T).T

y = [transformed_box[0][1], transformed_box[1][1], transformed_box[2][1], transformed_box[3][1]]
x = [transformed_box[0][0], transformed_box[1][0], transformed_box[2][0], transformed_box[3][0]]

y1, y2 = int(min(y)), int(max(y))
x1, x2 = int(min(x)), int(max(x))

# crop
crop = rotated[y1:y2, x1:x2]

# 흑백처리
gray2 = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

cv2.imwrite("mask.jpg", mask)
cv2.imwrite("box.jpg", src_box)
cv2.imwrite("canny.jpg", canned)
cv2.imwrite("rotated.jpg", rotated)
cv2.imwrite("cropped.jpg", crop)
cv2.imwrite("gray.jpg", gray2)