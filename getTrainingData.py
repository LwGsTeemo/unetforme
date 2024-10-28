import os
import shutil

# 設置來源和目標路徑
source_dir = '/home/Medical-SAM2/data/btcv/Training/image'  # 原始圖片資料夾
target_dir = '/home/UNet3plus_pth/data/train/imgs'  # 目標資料夾
npy_folder_path = '/home/UNet3plus_pth/masksRawData/'

def resizePic():
    import os
    from PIL import Image
    # 來源圖片的資料夾路徑
    source_dir = '/home/UNet3plus_pth/data/train/masks/'
    # 調整大小後圖片的保存資料夾路徑
    target_dir = '/home/UNet3plus_pth/data600x800/train/masks/'

    # 如果目標資料夾不存在，則創建
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 遍歷來源資料夾中的所有圖片
    for file_name in os.listdir(source_dir):
        file_path = os.path.join(source_dir, file_name)
        
        # 確保是圖片文件（而不是資料夾）
        if os.path.isfile(file_path):
            try:
                # 讀取圖片
                image = Image.open(file_path)
                
                # 調整大小到 369x369
                resized_image = image.resize((600, 800))
                
                # 構建保存路徑
                save_path = os.path.join(target_dir, file_name)
                
                # 保存調整大小後的圖片
                resized_image.save(save_path)
                print(f'已成功處理: {file_name}')
            except Exception as e:
                print(f'處理圖片 {file_name} 時發生錯誤: {e}')

    print(f'所有圖片已完成處理並保存在 {target_dir} 中。')


def seeNpyPic(npy_folder_path):
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    # np.set_printoptions(threshold=np.inf)
    for i in range(1,17232,1):
        data = np.load(f'{npy_folder_path}/{i:05d}_matte.npy')
        print(i)
        plt.imshow(data, cmap='gray')
        plt.axis('off')
        plt.savefig(f'/home/UNet3plus_pth/data/train/masks/{i:05d}_matte.jpg', bbox_inches='tight', pad_inches=0)
        plt.clf()

def mvData(source_dir,target_dir):
    # 如果目標資料夾不存在，創建它
    # if not os.path.exists(target_dir):
    #     os.makedirs(target_dir)

    # 設置文件計數器
    file_counter = 1

    # 遍歷 image0 到 image100 資料夾
    for folder_num in range(9,83,1):  # image0 到 image100 (共101個資料夾)
        folder_path = os.path.join(source_dir, f'img{folder_num:04d}')
        
        # 確保資料夾存在
        if os.path.exists(folder_path):
            # 遍歷資料夾中的所有圖片文件
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                
                # 檢查是否是文件而非資料夾
                if os.path.isfile(file_path):
                    # 構建新文件名，按順序編號
                    new_file_name = f'{file_counter:05d}.jpg'  # 例如：image_00001.jpg
                    new_file_path = os.path.join(target_dir, new_file_name)
                    
                    # 移動並重命名文件
                    shutil.move(file_path, new_file_path)
                    
                    # 更新計數器
                    file_counter += 1
    print("done.")

def rgb2Black(source_dir,target_dir):
    import os
    from PIL import Image

    # 設置來源圖片資料夾和目標資料夾
    source_dir = source_dir  # 替換為你的來源圖片資料夾
    target_dir = target_dir  # 替換為你的目標資料夾

    # 如果目標資料夾不存在，則創建
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 遍歷來源資料夾中的所有圖片
    for file_name in os.listdir(source_dir):
        file_path = os.path.join(source_dir, file_name)
        
        # 確保是圖片文件（而不是資料夾）
        if os.path.isfile(file_path):
            try:
                # 讀取彩色圖片
                image = Image.open(file_path)
                
                # 將圖片轉換為黑白（灰度）
                bw_image = image.convert('L')
                
                # 構建保存路徑
                save_path = os.path.join(target_dir, f'{file_name}')  # 添加 'bw_' 前綴以區分
                
                # 保存黑白圖片
                bw_image.save(save_path)
                print(f'已成功處理: {file_name}')
            except Exception as e:
                print(f'處理圖片 {file_name} 時發生錯誤: {e}')

    print(f'所有圖片已完成處理並保存在 {target_dir} 中。')


if __name__ == '__main__':
    # seeNpyPic(npy_folder_path)
    # mvData(source_dir,target_dir)
    resizePic()
    # rgb2Black(source_dir,target_dir)

    # from PIL import Image
    # # 讀取圖片
    # image = Image.open('/home/UNet3plus_pth/data/train/masks/00041_matte.jpg')  # 替換為你的圖片路徑
    # # image = Image.open('/home/UNet3plus_pth/dataSample/train/masks/00001_matte.png')

    # # 獲取圖片的層數（通道數）
    # num_layers = len(image.getbands())
    # width, height = image.size
    # # 輸出圖片的層數
    # print(f'圖片的層數（通道數）: {num_layers}')
    # print(f'圖片的大小: {width}x{height}')

