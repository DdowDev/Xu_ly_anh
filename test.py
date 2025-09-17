import cv2
import numpy as np
from matplotlib import pyplot as plt

# Đọc ảnh (ảnh nhị phân hoặc xám)
img = cv2.imread('elon.png', 0)  # nhớ để cùng thư mục hoặc thay đường dẫn

# Chuyển ảnh sang nhị phân (threshold)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Tạo structuring element (kernel)
kernel = np.ones((5, 5), np.uint8)

# Erosion
erosion = cv2.erode(binary, kernel, iterations=1)

# Dilation
dilation = cv2.dilate(binary, kernel, iterations=1)

# Opening (erosion rồi dilation)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

# Closing (dilation rồi erosion)
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# Morphological Gradient (biên)
gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)

# Hiển thị kết quả
titles = ['Original', 'Binary', 'Erosion', 'Dilation', 'Opening', 'Closing', 'Gradient']
images = [img, binary, erosion, dilation, opening, closing, gradient]

plt.figure(figsize=(12, 8))
for i in range(len(images)):
    plt.subplot(2, 4, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
