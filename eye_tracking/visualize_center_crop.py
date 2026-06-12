"""
用來查看stage1裡的第一次靜態裁切的範圍（中央聚焦區域）
"""

import cv2
import numpy as np
from pathlib import Path

def visualize_center_crop(image_path: str, 
                         target_center_w: int = 1600, 
                         target_center_h: int = 960):
    """
    視覺化第一次靜態裁切的中央區域
    
    Args:
        image_path: 測試圖片路徑
        target_center_w: 中央裁切寬度
        target_center_h: 中央裁切高度
    """
    # 讀取圖像
    image = cv2.imread(image_path)
    if image is None:
        print(f" 無法讀取圖像: {image_path}")
        return
    
    img_h, img_w = image.shape[:2]
    print(f"原始圖像尺寸: {img_w} x {img_h}")
    
    # 創建視覺化圖像
    vis_image = image.copy()
    
    # 計算中央裁切範圍（與 stage1_face_detection.py 完全相同）
    if img_w > target_center_w and img_h > target_center_h:
        center_x_offset = (img_w - target_center_w) // 2
        center_y_offset = (img_h - target_center_h) // 2
        
        center_x_max = center_x_offset + target_center_w
        center_y_max = center_y_offset + target_center_h
        
        # 繪製紅色矩形框標示中央裁切區域
        cv2.rectangle(vis_image, 
                     (center_x_offset, center_y_offset),
                     (center_x_max, center_y_max),
                     (0, 0, 255), 3)  # 紅色，線寬 3
        
        # 添加文字標註
        cv2.putText(vis_image, 
                   f'Center Crop: {target_center_w}x{target_center_h}',
                   (center_x_offset + 10, center_y_offset + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        
        # 添加半透明遮罩到裁切區域外
        mask = np.zeros_like(image, dtype=np.uint8)
        mask[center_y_offset:center_y_max, center_x_offset:center_x_max] = 255
        
        # 創建暗化效果
        darkened = (vis_image * 0.4).astype(np.uint8)
        vis_image = np.where(mask == 255, vis_image, darkened)
        
        # 重新繪製邊框（因為可能被遮罩覆蓋）
        cv2.rectangle(vis_image, 
                     (center_x_offset, center_y_offset),
                     (center_x_max, center_y_max),
                     (0, 0, 255), 3)
        
        print(f" 中央裁切範圍: x=[{center_x_offset}, {center_x_max}], y=[{center_y_offset}, {center_y_max}]")
        print(f"   尺寸: {target_center_w} x {target_center_h}")
    else:
        # 圖像太小，不需要裁切
        cv2.putText(vis_image, 
                   'No center crop (image too small)',
                   (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        print("⚠️ 圖像尺寸小於目標裁切尺寸，不執行中央裁切")
    
    # 保存結果
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'center_crop_visualization.jpg'
    cv2.imwrite(str(output_path), vis_image)
    print(f" 視覺化結果已保存: {output_path}")
    
    # 顯示圖像（可選）
    cv2.imshow('Center Crop Visualization (Press any key to close)', vis_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='視覺化第一次靜態裁切範圍')
    parser.add_argument('--input', type=str, required=True, help='測試圖片路徑')
    parser.add_argument('--width', type=int, default=1600, help='中央裁切寬度')
    parser.add_argument('--height', type=int, default=960, help='中央裁切高度')
    
    args = parser.parse_args()
    
    visualize_center_crop(args.input, args.width, args.height)
