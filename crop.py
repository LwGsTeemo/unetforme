def changeNpyToJpg():
    import os
    import numpy as np
    from PIL import Image

    # 設定包含 .npy 檔案的資料夾路徑
    folder_path = '/home/Medical-SAM2/data/btcv_btcv/Training/mask'

    # 遍歷資料夾內的所有 .npy 檔案
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.npy'):
            file_path = os.path.join(folder_path, file_name)
            
            # 載入 .npy 檔案
            data = np.load(file_path)
            
            # 將數據範圍標準化到 0-255，轉為 8 位無符號整數
            data_normalized = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)
            
            # 轉換為灰階圖片
            img = Image.fromarray(data_normalized, mode='L')
            
            # 二值化處理（假設閾值為 128，可以根據需要調整）
            threshold = 128
            img_binary = img.point(lambda p: 255 if p > threshold else 0)

            # 存回新檔案，使用相同檔名但副檔名改為 .jpg
            new_path = '/home/Medical-SAM2/data/btcv_btcv/Training/afterMask'
            output_path = os.path.join(new_path, file_name.replace('.npy', '.jpg'))
            img_binary.save(output_path)

    print("所有 .npy 檔案已轉換並二值化完成！")

def seeContourInPic():
    import cv2
    import numpy as np
    import os
    # 指定資料夾
    contour_folder = '/home/UNet3plus_pth/dataFew/output/mask/'
    train_folder = '/home/UNet3plus_pth/dataFew/output/image/'
    output_folder = '/home/UNet3plus_pth/dataFew/output/369x369/'

    # 確保輸出資料夾存在
    os.makedirs(output_folder, exist_ok=True)

    # 讀取輪廓資料夾中的所有 .jpg 檔案名稱
    contour_files = [f for f in os.listdir(contour_folder) if f.endswith('.jpg')]

    for contour_file in contour_files:
        # 建立對應的訓練資料和輸出路徑
        contour_path = os.path.join(contour_folder, contour_file)
        train_path = os.path.join(train_folder, f'{contour_file}') #對名字
        output_path = os.path.join(output_folder, contour_file.replace('.jpg', '.jpg'))
        
        # 確認訓練資料圖片是否存在
        if not os.path.exists(train_path):
            print(f"對應的訓練圖片不存在: {train_path}")
            continue

        # 讀取輪廓和訓練圖片
        contour_image = cv2.imread(contour_path, cv2.IMREAD_GRAYSCALE)
        train_image = cv2.imread(train_path, cv2.IMREAD_GRAYSCALE)

        # 確保尺寸一致
        contour_resized = cv2.resize(contour_image, (train_image.shape[1], train_image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 將輪廓轉為二值圖像
        _, binary_contour = cv2.threshold(contour_resized, 240, 255, cv2.THRESH_BINARY)

        # 找到輪廓
        contours, _ = cv2.findContours(binary_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 在訓練圖片上疊加輪廓
        overlay = train_image.copy()
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)  # 紅色輪廓

        # 調整透明度並產生結果
        alpha = 0.5
        result = cv2.addWeighted(overlay, alpha, train_image, 1 - alpha, 0)

        # 儲存疊加結果
        cv2.imwrite(output_path, result)
        print(f"已儲存疊加結果: {output_path}")



if __name__ == '__main__':
    # seeContourInPic()
    # import cv2
    # import numpy as np

    # # 讀取標籤輪廓和訓練灰階圖片
    # label_img = cv2.imread('/home/UNet3plus_pth/data/Training/afterMask/5.jpg', cv2.IMREAD_GRAYSCALE)
    # train_img = cv2.imread('/home/UNet3plus_pth/data/Training/afterImage/5.jpg', cv2.IMREAD_GRAYSCALE)

    # # 將標籤圖片調整到訓練圖片的大小
    # label_resized = cv2.resize(label_img, (train_img.shape[1], train_img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # # 將標籤輪廓二值化，使其更明顯
    # _, label_binary = cv2.threshold(label_resized, 128, 255, cv2.THRESH_BINARY)

    # # 疊加圖片，設置透明度
    # alpha = 0.6  # 訓練圖片的透明度
    # beta = 0.4   # 標籤輪廓的透明度
    # overlay_img = cv2.addWeighted(train_img, alpha, label_binary, beta, 0)

    # # 保存或顯示結果
    # cv2.imwrite('/home/UNet3plus_pth/data/Training/5.jpg', overlay_img)
    # print("圖片重疊完成並已保存！")

