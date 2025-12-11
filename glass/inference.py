import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glass.glass import GLASS
import backbones
from torchvision import transforms
from PIL import Image

# ================= 設定區 =================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BACKBONE_NAME = 'wideresnet50'
# 這是你訓練好的權重路徑 (請換成你實際的路徑)
CKPT_PATH = './results/membank_5shot/models/backbone_0/mvtec_transistor/ckpt_best_2329.pth'
# 這是你要測試的圖片
TEST_IMG_PATH = 'datasets/mvtec_ad_dataset/transistor/test/bent_lead/001.png'
# 這是那 5 張 Support Set (正常) 圖片的路徑 (為了初始化 Memory Bank)
# 注意：這裡必須換成你訓練時用的那 5 張圖
SUPPORT_IMG_PATHS = [
    'datasets/mvtec_ad_dataset/transistor/train/good/000.png',
    'datasets/mvtec_ad_dataset/transistor/train/good/001.png',
    'datasets/mvtec_ad_dataset/transistor/train/good/002.png',
    'datasets/mvtec_ad_dataset/transistor/train/good/003.png',
    'datasets/mvtec_ad_dataset/transistor/train/good/004.png',
]
# =========================================

def load_and_transform(path, size=288):
    """讀取圖片並轉成 Tensor"""
    img = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0) # 變成 (1, 3, H, W)

def main():
    print(f"Loading model on {DEVICE}...")
    
    # 1. 建立模型架構 (參數要跟訓練時一樣)
    backbone = backbones.load(BACKBONE_NAME)
    model = GLASS(DEVICE)
    model.load(
        backbone=backbone,
        layers_to_extract_from=['layer2', 'layer3'],
        device=DEVICE,
        input_shape=(3, 288, 288),
        pretrain_embed_dimension=1536,
        target_embed_dimension=1536,
        memory_bank_size=5 # 設定 Bank 大小
    )
    
    # 2. 載入權重
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    # 處理一下 key 的名稱 (有時候會有 module. 前綴)
    if 'discriminator' in ckpt:
        model.discriminator.load_state_dict(ckpt['discriminator'])
    if 'pre_projection' in ckpt and model.pre_proj > 0:
        model.pre_projection.load_state_dict(ckpt['pre_projection'])
    
    model.to(DEVICE)
    model.eval()
    
    # 3. [關鍵] 初始化 Memory Bank
    # 因為 Memory Bank 的內容沒有存在權重檔裡，必須重新計算
    print("Initializing Memory Bank with support set...")
    support_tensors = [load_and_transform(p) for p in SUPPORT_IMG_PATHS]
    support_batch = torch.cat(support_tensors, dim=0).to(DEVICE) # (5, 3, 288, 288)
    model.init_memory_bank(support_batch)
    
    # 4. 進行推論
    print(f"Testing image: {TEST_IMG_PATH}")
    test_tensor = load_and_transform(TEST_IMG_PATH).to(DEVICE)
    
    # 使用 _predict 取得分數
    scores, masks = model._predict(test_tensor)
    anomaly_map = masks[0] # (288, 288) numpy array
    
    # 5. 視覺化繪圖
    original_img = Image.open(TEST_IMG_PATH).resize((288, 288))
    
    plt.figure(figsize=(10, 5))
    
    # 左圖：原圖
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title("Original Image")
    plt.axis('off')
    
    # 右圖：異常熱力圖
    plt.subplot(1, 2, 2)
    plt.imshow(original_img)
    plt.imshow(anomaly_map, cmap='jet', alpha=0.5) # alpha=0.5 讓圖變半透明疊加
    plt.title(f"Anomaly Map (Max Score: {np.max(anomaly_map):.2f})")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('inference_result.png')
    print("Result saved to inference_result.png")
    plt.show()

if __name__ == '__main__':
    main()