"""
第四階段：神經網絡推理
Stage 4: Neural Network Inference

使用 ETH-XGaze 預訓練的 ResNet-50 模型進行視線估計：
- 載入預訓練模型
- 圖像預處理（正規化）
- 推理得到視線角度（pitch, yaw）
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import numpy as np
import cv2
from typing import Dict, Optional, Tuple
from pathlib import Path


class GazeNetwork(nn.Module):
    """
    基於 ResNet-50 的視線估計網絡
    輸入：224×224×3 RGB 圖像
    輸出：2D 視線角度（pitch, yaw）
    
    這個架構與 ETH-XGaze 預訓練模型完全匹配
    """
    
    def __init__(self, pretrained=False):
        """
        初始化視線網絡
        
        Args:
            pretrained: 是否使用 ImageNet 預訓練權重（僅在訓練時使用）
        """
        super(GazeNetwork, self).__init__()
        
        # 使用 ResNet-50 作為骨幹網絡
        # 但要去掉最後的分類層（fc）
        resnet = models.resnet50(pretrained=pretrained)
        
        # 複製 ResNet-50 的結構（不使用Sequential，保留完整結構）
        # 這樣鍵名會是 gaze_network.conv1, gaze_network.layer1, etc.
        self.gaze_network = nn.Module()
        self.gaze_network.conv1 = resnet.conv1
        self.gaze_network.bn1 = resnet.bn1
        self.gaze_network.relu = resnet.relu
        self.gaze_network.maxpool = resnet.maxpool
        self.gaze_network.layer1 = resnet.layer1
        self.gaze_network.layer2 = resnet.layer2
        self.gaze_network.layer3 = resnet.layer3
        self.gaze_network.layer4 = resnet.layer4
        self.gaze_network.avgpool = resnet.avgpool
        
        # 視線估計的全連接層（從 2048 特徵到 2D 視線）
        self.gaze_fc = nn.Sequential(
            nn.Linear(2048, 2)  # 直接從特徵到視線角度
        )
    
    def forward(self, x):
        """
        前向傳播
        
        Args:
            x: 輸入圖像張量 (batch_size, 3, 224, 224)
        
        Returns:
            gaze: 視線角度 (batch_size, 2) - [pitch, yaw] 弧度
        """
        # ResNet-50 特徵提取
        x = self.gaze_network.conv1(x)
        x = self.gaze_network.bn1(x)
        x = self.gaze_network.relu(x)
        x = self.gaze_network.maxpool(x)
        
        x = self.gaze_network.layer1(x)
        x = self.gaze_network.layer2(x)
        x = self.gaze_network.layer3(x)
        x = self.gaze_network.layer4(x)
        
        x = self.gaze_network.avgpool(x)
        feature = x.view(x.size(0), -1)  # (batch_size, 2048)
        
        # 直接使用 gaze_fc 進行視線估計
        # 不使用 gaze_network.fc（那是 ImageNet 分類層）
        gaze = self.gaze_fc(feature)
        
        return gaze


class GazeEstimator:
    """
    視線估計器
    
    載入預訓練模型並進行推理
    """
    
    def __init__(self, config: Dict = None):
        """
        初始化視線估計器
        
        Args:
            config: 配置字典，包含以下參數：
                - model_path: 預訓練模型路徑
                - device: 計算設備 ('cuda' 或 'cpu')
                - use_gpu: 是否使用 GPU（向後兼容）
        """
        if config is None:
            config = {}
        
        # 模型路徑
        self.model_path = config.get('model_path', 'models/epoch_24_ckpt.pth.tar')
        
        # 設備設置
        use_gpu = config.get('use_gpu', True)
        device_name = config.get('device', 'cuda' if use_gpu else 'cpu')
        
        # 檢查 GPU 可用性
        if device_name == 'cuda' and not torch.cuda.is_available():
            print("警告: CUDA 不可用，使用 CPU")
            device_name = 'cpu'
        
        self.device = torch.device(device_name)
        
        # 圖像預處理轉換
        # ImageNet 標準正規化參數
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),  # 將像素值從 [0,255] 轉為 [0,1]
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet 均值
                std=[0.229, 0.224, 0.225]     # ImageNet 標準差
            ),
        ])
        
        # 載入模型
        self.model = self._load_model()
        
        print(f"✓ GazeEstimator 初始化完成")
        print(f"  - 模型路徑: {self.model_path}")
        print(f"  - 設備: {self.device}")
        print(f"  - 模型已載入")
    
    def _load_model(self) -> nn.Module:
        """
        載入預訓練的視線估計模型 (修正版：加入前綴過濾邏輯)
        """
        # 1. 檢查模型文件是否存在
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"找不到預訓練模型: {self.model_path}")
        
        # 2. 創建模型實例
        model = GazeNetwork(pretrained=False)
        
        try:
            # 3. 載入原始 checkpoint
            if self.device.type == 'cuda':
                checkpoint = torch.load(self.model_path)
            else:
                checkpoint = torch.load(self.model_path, map_location='cpu')
            
            # --- 核心修正區塊：加入前綴過濾與結構適配 ---
            # 從 checkpoint 中提取真正的權重字典
            if 'model_state' in checkpoint:
                state_dict = checkpoint['model_state']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # 重要：過濾 'module.' 前綴並確保鍵名匹配
            # 這是因為官方模型若在 DataParallel 下訓練，鍵名會多出 module.
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            # 4. 載入過濾後的權重
            # 這樣即使模型裡沒有定義 fc，載入時也不會因為多出鍵值而報錯
            keys_to_remove = ['gaze_network.fc.weight', 'gaze_network.fc.bias']
            for key in keys_to_remove:
                if key in new_state_dict:
                    del new_state_dict[key]
                    print(f"  - 已剔除冗餘權重鍵值: {key}")

            # 3. 執行載入
            model.load_state_dict(new_state_dict, strict=True) 
            print(f"✓ 模型權重載入成功 (已精簡結構)")
            # ----------------------------------------
            
        except Exception as e:
            raise RuntimeError(f"載入模型權重失敗: {e}")
        
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        預處理輸入圖像
        
        Args:
            image: 輸入圖像 (224, 224, 3) BGR 格式
        
        Returns:
            tensor: 預處理後的張量 (1, 3, 224, 224)
        """
        # BGR 轉 RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 應用轉換
        tensor = self.transform(image_rgb)
        
        # 添加批次維度
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def estimate(self, normalized_image: np.ndarray) -> Dict:
        """
        估計視線方向
        
        Args:
            normalized_image: 正規化後的人臉圖像 (224, 224, 3) - 來自第三階段
        
        Returns:
            result: 包含視線估計結果的字典：
                - 'gaze_angles': 視線角度 [pitch, yaw]（弧度）
                - 'gaze_angles_deg': 視線角度 [pitch, yaw]（度）
                - 'success': 是否成功
        """
        try:
            # 1. 核心修正：將 448x448 強制縮放至 224x224
            if normalized_image.shape[:2] != (224, 224):
                normalized_image = cv2.resize(normalized_image, (224, 224), 
                                           interpolation=cv2.INTER_AREA)
            
            # 2. 數據預處理 (ToTensor + Normalize 已在 __init__ 定義)
            input_tensor = self.preprocess_image(normalized_image).to(self.device)
            
            # 3. 模型推理
            with torch.no_grad():
                output = self.model(input_tensor)
            
            gaze_angles = output.cpu().numpy()[0]  # [pitch, yaw]
            
            # 轉換為 numpy 數組
            gaze_angles = output.cpu().numpy()[0]  # [pitch, yaw] 弧度
            
            # 轉換為度
            gaze_angles_deg = np.rad2deg(gaze_angles)
            
            return {
                'gaze_angles': gaze_angles,           # 弧度
                'gaze_angles_deg': gaze_angles_deg,   # 度
                'success': True
            }
        
        except Exception as e:
            print(f"錯誤: 視線估計失敗 - {e}")
            return {
                'gaze_angles': np.zeros(2),
                'gaze_angles_deg': np.zeros(2),
                'success': False
            }
    
    def batch_estimate(self, normalized_images: list) -> list:
        """
        批量估計多張圖像的視線方向
        
        Args:
            normalized_images: 正規化圖像列表
        
        Returns:
            results: 估計結果列表
        """
        results = []
        
        # 預處理所有圖像
        tensors = []
        for img in normalized_images:
            tensor = self.preprocess_image(img)
            tensors.append(tensor)
        
        # 合併為批次
        batch_tensor = torch.cat(tensors, dim=0).to(self.device)
        
        # 批次推理
        with torch.no_grad():
            outputs = self.model(batch_tensor)
        
        # 轉換結果
        outputs_np = outputs.cpu().numpy()
        
        for gaze_angles in outputs_np:
            gaze_angles_deg = np.rad2deg(gaze_angles)
            results.append({
                'gaze_angles': gaze_angles,
                'gaze_angles_deg': gaze_angles_deg,
                'success': True
            })
        
        return results
    
    def draw_gaze_on_image(self, image, pitch, yaw, length=100):
        # 1. 從角度轉回 3D 向量 (與官方 vector_to_pitchyaw 逆運算一致)
        dx = -np.cos(pitch) * np.sin(yaw)
        dy = -np.sin(pitch)
        # 如果 Z 軸朝向相機，則 dz 為負
        
        # 2. 繪製箭頭 (pos 為畫面中心 112, 112)
        center = (image.shape[1] // 2, image.shape[0] // 2)
        end_point = (int(center[0] + dx * length), int(center[1] + dy * length))
        
        cv2.arrowedLine(image, center, end_point, (0, 0, 255), 2, cv2.LINE_AA)
        return image


# ==================== 測試和示例代碼 ====================

def test_gaze_estimator():
    """
    測試視線估計器
    """
    print("=" * 70)
    print("測試第四階段：神經網絡推理")
    print("=" * 70)
    
    # 創建估計器
    config = {
        'model_path': 'models/epoch_24_ckpt.pth.tar',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    try:
        estimator = GazeEstimator(config=config)
    except FileNotFoundError as e:
        print(f"\n錯誤: {e}")
        print("\n請確保預訓練模型文件存在:")
        print("  - models/epoch_24_ckpt.pth.tar")
        return
    except Exception as e:
        print(f"\n錯誤: 初始化失敗 - {e}")
        return
    
    # 創建模擬的正規化圖像（224×224×3）
    print("\n創建模擬正規化圖像...")
    normalized_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # 繪製一個簡單的人臉（用於可視化）
    cv2.circle(normalized_image, (112, 90), 20, (255, 200, 150), -1)  # 臉
    cv2.circle(normalized_image, (90, 85), 5, (0, 0, 0), -1)   # 左眼
    cv2.circle(normalized_image, (134, 85), 5, (0, 0, 0), -1)  # 右眼
    cv2.ellipse(normalized_image, (112, 110), (15, 8), 0, 0, 180, (0, 0, 0), 2)  # 嘴
    
    # 執行視線估計
    print("\n執行視線估計...")
    result = estimator.estimate(normalized_image)
    
    if result['success']:
        print("✓ 視線估計成功！")
        print(f"\n視線角度（弧度）:")
        print(f"  Pitch: {result['gaze_angles'][0]:>7.4f} rad")
        print(f"  Yaw:   {result['gaze_angles'][1]:>7.4f} rad")
        print(f"\n視線角度（度）:")
        print(f"  Pitch: {result['gaze_angles_deg'][0]:>7.2f}°")
        print(f"  Yaw:   {result['gaze_angles_deg'][1]:>7.2f}°")
        
        # 繪製視線方向
        print("\n生成可視化結果...")
        vis_image = estimator.draw_gaze_on_image(
            normalized_image,
            pitch=result['gaze_angles'][0],
            yaw=result['gaze_angles'][1],
            thickness=3
        )
        
        # 添加文字
        cv2.putText(vis_image, 
                   f"Pitch: {result['gaze_angles_deg'][0]:.1f}deg", 
                   (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(vis_image, 
                   f"Yaw: {result['gaze_angles_deg'][1]:.1f}deg", 
                   (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 保存結果
        from pathlib import Path
        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)
        
        cv2.imwrite(str(output_dir / 'test_gaze_estimation.jpg'), vis_image)
        print(f"✓ 結果已保存到 output/test_gaze_estimation.jpg")
    else:
        print("❌ 視線估計失敗")
    
    print("\n" + "=" * 70)
    print("測試完成！")
    print("=" * 70)


if __name__ == '__main__':
    test_gaze_estimator()

