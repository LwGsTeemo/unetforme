image_folder = '/home/UNet3plus_pth/data/train/imgs/'
output_folder = '/home/UNet3plus_pth/data/train/imgs2/'

def rotatePicture(image_folder,output_folder):
    import cv2
    import os

    # 確保輸出資料夾存在
    os.makedirs(output_folder, exist_ok=True)

    # 讀取資料夾中的所有圖片檔案
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        output_path = os.path.join(output_folder, image_file)

        # 讀取圖片
        image = cv2.imread(image_path)

        # 旋轉圖片90度 (順時針)
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # 儲存旋轉後的圖片
        cv2.imwrite(output_path, rotated_image)
        print(f"已旋轉並儲存圖片: {output_path}")

def prepareCropPic():
    import os
    import cv2
    import numpy as np

    # 設定包含灰階圖片和對應彩色圖片的資料夾路徑
    # folder1_path = '/home/UNet3plus_pth/data/Training/afterImage'
    # folder2_path = '/home/UNet3plus_pth/data/Training/afterMask'
    # output_folder1 = '/home/UNet3plus_pth/data/cropTraining/image'
    # output_folder2 = '/home/UNet3plus_pth/data/cropTraining/mask'
    folder1_path = '/home/UNet3plus_pth/dataFew/train/imgs'
    folder2_path = '/home/UNet3plus_pth/dataFew/train/masks'
    output_folder1 = '/home/UNet3plus_pth/dataFew/output/image'
    output_folder2 = '/home/UNet3plus_pth/dataFew/output/mask'

    # 確保輸出資料夾存在
    os.makedirs(output_folder1, exist_ok=True)
    os.makedirs(output_folder2, exist_ok=True)

    # 逐一處理第一組資料夾中的每張圖片
    for file_name in os.listdir(folder1_path):
        if file_name.endswith('.jpg'):
            # 設定兩組圖片的路徑
            img1_path = os.path.join(folder1_path, file_name)
            img2_path = os.path.join(folder2_path, file_name)  # 假設兩組圖片名稱相同

            # 確認對應的圖片存在
            if not os.path.exists(img2_path):
                print(f"對應的第二組圖片 {img2_path} 不存在，跳過該組圖片")
                continue

            # 讀取兩組灰階圖片
            img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

            # 使用 Canny 邊緣檢測
            edges = cv2.Canny(img1, threshold1=100, threshold2=200)

            # 應用高斯模糊以減少背景細節
            blurred = cv2.GaussianBlur(img1, (15, 15), sigmaX=0)

            # 創建一個空白的圖片來顯示強化效果
            enhanced_image = np.zeros_like(img1)

            # 使用邊緣作為掩膜將物體邊緣強化
            enhanced_image[:, :,] = np.where(edges == 255, img1[:, :,], blurred[:, :,])

            # 二值化處理，便於偵測輪廓
            _, binary_img = cv2.threshold(enhanced_image, 240, 255, cv2.THRESH_BINARY_INV)
            # 找出所有輪廓並選取最大輪廓
            contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 確保有找到輪廓
            if contours:
                # 找出最大輪廓
                max_contour = max(contours, key=cv2.contourArea)

                # 獲取最大輪廓的邊界框
                x, y, w, h = cv2.boundingRect(max_contour)

                # 擷取出最大輪廓的區域
                cropped_region1 = img1[y:y+h, x:x+w]
                cropped_region2 = img2[y:y+h, x:x+w]
            # contour_img = np.zeros_like(img1)
            # # 畫出最大輪廓，顏色為白色，線條粗細為2
            # cv2.drawContours(contour_img, [max_contour], -1, (255), thickness=2)

            # 保存處理後的圖片
            output_img1_path = os.path.join(output_folder1, file_name)
            output_img2_path = os.path.join(output_folder2, file_name)
            cv2.imwrite(output_img1_path, cropped_region1)
            cv2.imwrite(output_img2_path, cropped_region2)

    print("所有圖片的區域提取完成並已保存！")

def resizeLabelSize():
    import cv2
    import os
    # 定義圖片和標籤資料夾的路徑
    train_img_folder = '/home/UNet3plus_pth/data/Training/afterImage'
    label_img_folder = '/home/UNet3plus_pth/data/Training/afterMask'
    output_folder = '/home/UNet3plus_pth/data/Training/resizeLabelSize'

    os.makedirs(output_folder, exist_ok=True)
    # 遍歷標籤資料夾中的所有圖片
    for filename in os.listdir(label_img_folder):
        if filename.endswith('.jpg'):
            # 構建標籤圖片和訓練圖片的路徑
            label_path = os.path.join(label_img_folder, filename)
            train_path = os.path.join(train_img_folder, filename)

            # 讀取標籤和訓練圖片
            label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            train_img = cv2.imread(train_path, cv2.IMREAD_GRAYSCALE)

            # 確保讀取圖片成功
            if label_img is None or train_img is None:
                print(f"無法讀取圖片 {filename}，跳過此檔案。")
                continue

            # 將標籤圖片調整到訓練圖片的大小
            label_resized = cv2.resize(label_img, (train_img.shape[1], train_img.shape[0]), interpolation=cv2.INTER_NEAREST)

            # 設定新路徑並保存調整後的標籤圖片到指定的輸出資料夾
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, label_resized)
            print(f"{filename} 已成功調整並保存到 {output_path}。")

    print("所有圖片處理完成！")

def calIOU():
    import cv2
    import numpy as np
    import os

    # 定義兩個資料夾的路徑
    folder1 = '/home/UNet3plus_pth/data/Training/afterImage'  # 第一個資料夾
    folder2 = '/home/UNet3plus_pth/data/Training/resizeLabelSize'  # 第二個資料夾
    large = 0
    # 確保兩個資料夾中的檔案數量和名稱一致
    for filename in os.listdir(folder1):
        if filename.endswith('.jpg'):
            # 構建兩個檔案的完整路徑
            path1 = os.path.join(folder1, filename)
            path2 = os.path.join(folder2, filename)

            # 讀取兩張灰階圖片
            img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

            # 確保圖片大小一致
            if img1.shape != img2.shape:
                print(f"圖片 {filename} 的大小不一致，跳過此檔案。")
                continue

            # 將圖片二值化（可根據需要調整閾值）
            _, binary_img1 = cv2.threshold(img1, 240, 255, cv2.THRESH_BINARY_INV)
            _, binary_img2 = cv2.threshold(img2, 128, 255, cv2.THRESH_BINARY)

            # 找出二值圖中的輪廓
            contours1, _ = cv2.findContours(binary_img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours2, _ = cv2.findContours(binary_img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 建立空白遮罩來填充輪廓
            mask1 = np.zeros_like(binary_img1)
            mask2 = np.zeros_like(binary_img2)

            # 將輪廓填充到遮罩中
            cv2.drawContours(mask1, contours1, -1, 255, thickness=cv2.FILLED)
            cv2.drawContours(mask2, contours2, -1, 255, thickness=cv2.FILLED)

            # 計算交集和聯集
            intersection = np.logical_and(mask1, mask2).sum()
            union = np.logical_or(mask1, mask2).sum()

            # 計算 IOU
            iou = intersection / union if union != 0 else 0
            large = max(large,iou)
            print(f"{filename} 的 IoU 為: {iou}")

    print("所有同名圖片的 IoU 計算完成！",large)



if __name__ == '__main__':
    # rotatePicture(image_folder,output_folder)
    # prepareCropPic()
    # resizeLabelSize()
    calIOU()