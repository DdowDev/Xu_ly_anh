import cv2
import numpy as np
import matplotlib.pyplot as plt

# Tạo ảnh đen 200x200
img = np.zeros((200, 200), dtype=np.uint8)

# Vẽ hình vuông trắng ở giữa (50x50)
cv2.rectangle(img, (75, 75), (125, 125), 255, -1)

# Thêm noise (các điểm trắng ngẫu nhiên bên ngoài vuông)
for _ in range(100):
    x, y = np.random.randint(0, 200), np.random.randint(0, 200)
    img[x, y] = 255

# Kernel 3x3
kernel = np.ones((3, 3), np.uint8)

# Các phép toán hình thái học
erosion = cv2.erode(img, kernel, iterations=1)
dilation = cv2.dilate(img, kernel, iterations=1)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

# Hiển thị kết quả
titles = ['Original (Square+Noise)', 'Erosion', 'Dilation', 'Opening', 'Closing', 'Gradient']
images = [img, erosion, dilation, opening, closing, gradient]

plt.figure(figsize=(10, 6))
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.show()
