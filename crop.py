import cv2
import numpy as np
import os

rgb_file = './cropping/00056.jpg'
gray_image = cv2.imread('./cropping/00056_matte.jpg', cv2.IMREAD_GRAYSCALE)

# 設定白色的閾值並找到白色部分
_, binary_image = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY)  # 閾值 240 視為白色
white_pixels = np.column_stack(np.where(binary_image == 255))

# 計算白色部分的群中心
if white_pixels.size > 0:
    center_y, center_x = np.mean(white_pixels, axis=0).astype(int)
else:
    raise ValueError("沒有找到白色區域")

# 讀取彩色圖片，這是初始的192x192x3圖片
color_image = cv2.imread(rgb_file)

# 設定裁剪區域的大小
crop_size = 200
half_crop_size = crop_size // 2

# 計算裁剪範圍
x_start = max(center_x - half_crop_size, 0)
y_start = max(center_y - half_crop_size, 0)
x_end = min(center_x + half_crop_size, color_image.shape[1])
y_end = min(center_y + half_crop_size, color_image.shape[0])

# 創建一個 192x192x3 的空白區域，初始為零（黑色）
cropped_image = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)

# 計算在新圖片中的位置
x_offset = max(half_crop_size - center_x, 0)
y_offset = max(half_crop_size - center_y, 0)

# 將彩色圖片的區域填入新的 192x192 圖片中
cropped_image[y_offset:y_offset + (y_end - y_start), x_offset:x_offset + (x_end - x_start)] = color_image[y_start:y_end, x_start:x_end]

# 儲存裁剪結果
file_name = os.path.basename(rgb_file).split('.')[0]
cv2.imwrite(f'./cropping/{file_name}_cropped_image.png', cropped_image)
