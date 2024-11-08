import cv2
import numpy as np

def calculate_iou_and_dice(image_path1, image_path2, threshold=128):
    # 讀取灰階圖片
    img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
    
    # 將圖片二值化（閾值默認為 128，若有需要可以調整）
    _, binary_img1 = cv2.threshold(img1, threshold, 1, cv2.THRESH_BINARY)
    _, binary_img2 = cv2.threshold(img2, threshold, 1, cv2.THRESH_BINARY)
    
    # 計算交集和並集
    intersection = np.logical_and(binary_img1, binary_img2).sum()
    union = np.logical_or(binary_img1, binary_img2).sum()
    
    # 計算 IOU 和 Dice 係數
    iou = intersection / union if union != 0 else 0
    dice = (2 * intersection) / (binary_img1.sum() + binary_img2.sum()) if (binary_img1.sum() + binary_img2.sum()) != 0 else 0
    
    return iou, dice

# 使用範例
image_path1 = "calIOU/00050_matte.jpg"
image_path2 = "calIOU/00056_matte.jpg"
iou, dice = calculate_iou_and_dice(image_path1, image_path2)
print("IOU:", iou)
print("Dice coefficient:", dice)
