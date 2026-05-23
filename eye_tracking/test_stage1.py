"""
測試第一階段：人臉檢測與特徵點定位
Test Stage 1: Face Detection and Landmark Localization

這個腳本用於測試 stage1_face_detection.py 的功能
"""

import cv2
import numpy as np
from pathlib import Path
from stages.stage1_face_detection import FaceDetector


def test_with_webcam():
    """使用網路攝像頭進行即時測試"""
    print("=" * 70)
    print("即時網路攝像頭測試")
    print("按 'q' 鍵退出")
    print("=" * 70)
    
    # 初始化檢測器
    detector = FaceDetector(config={'min_confidence': 0.5})
    
    # 打開攝像頭
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("錯誤: 無法打開攝像頭")
        return
    
    print("\n攝像頭已開啟，開始檢測...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 檢測人臉
        result = detector.detect(frame)
        
        if result is not None:
            # 可視化
            vis_frame = detector.visualize_landmarks(
                frame,
                landmarks_468=result['landmarks_468'],
                landmarks_selected=result['landmarks_2d_selected'],
                bbox=result['bbox'],
                show_all=False
            )
            
            # 顯示資訊
            info_text = f"Confidence: {result['confidence']:.3f}"
            cv2.putText(vis_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            vis_frame = frame
            cv2.putText(vis_frame, "No Face Detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 顯示畫面
        cv2.imshow('Face Detection Test', vis_frame)
        
        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n測試結束")


def test_with_image(image_path: str):
    """使用圖像文件進行測試"""
    print("=" * 70)
    print("圖像文件測試")
    print("=" * 70)
    
    # 讀取圖像
    image = cv2.imread(image_path)
    if image is None:
        print(f"錯誤: 無法讀取圖像 {image_path}")
        return
    
    print(f"\n讀取圖像: {image_path}")
    print(f"圖像尺寸: {image.shape[1]} x {image.shape[0]}")
    
    # 初始化檢測器
    detector = FaceDetector(config={'min_confidence': 0.5})
    
    # 檢測人臉
    print("\n執行檢測...")
    result = detector.detect(image)
    
    if result is None:
        print(" 未檢測到人臉")
        return
    
    print("✓ 檢測成功！")
    print(f"\n結果:")
    print(f"  - 468 個特徵點")
    print(f"  - {len(result['landmarks_2d_selected'])} 個關鍵點")
    print(f"  - 置信度: {result['confidence']:.3f}")
    
    # 顯示關鍵點座標
    print(f"\n關鍵點座標 (前 4 個):")
    for i in range(min(4, len(result['landmarks_2d_selected']))):
        point = result['landmarks_2d_selected'][i]
        print(f"  點 {i+1}: ({point[0]:.2f}, {point[1]:.2f})")
    
    # 可視化
    vis_image = detector.visualize_landmarks(
        image,
        landmarks_468=result['landmarks_468'],
        landmarks_selected=result['landmarks_2d_selected'],
        bbox=result['bbox'],
        show_all=False
    )
    
    # 保存結果
    output_path = Path('output/test_stage1_result.jpg')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis_image)
    print(f"\n✓ 結果已保存: {output_path}")
    
    # 顯示結果
    cv2.imshow('Detection Result', vis_image)
    print("\n按任意鍵關閉...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_batch_images(image_dir: str):
    """批量測試多張圖像"""
    print("=" * 70)
    print("批量圖像測試")
    print("=" * 70)
    
    image_dir = Path(image_dir)
    image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
    
    if not image_files:
        print(f"錯誤: 在 {image_dir} 中未找到圖像文件")
        return
    
    print(f"\n找到 {len(image_files)} 張圖像")
    
    # 初始化檢測器
    detector = FaceDetector(config={'min_confidence': 0.5})
    
    # 統計
    success_count = 0
    fail_count = 0
    
    # 處理每張圖像
    for img_path in image_files:
        print(f"\n處理: {img_path.name}")
        
        image = cv2.imread(str(img_path))
        if image is None:
            print("   無法讀取")
            fail_count += 1
            continue
        
        result = detector.detect(image)
        
        if result is not None:
            print(f"  ✓ 檢測成功 (置信度: {result['confidence']:.3f})")
            success_count += 1
            
            # 可視化並保存
            vis_image = detector.visualize_landmarks(
                image,
                landmarks_selected=result['landmarks_2d_selected'],
                bbox=result['bbox']
            )
            
            output_path = Path(f'output/batch_results/{img_path.name}')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), vis_image)
        else:
            print("   未檢測到人臉")
            fail_count += 1
    
    print("\n" + "=" * 70)
    print("批量測試完成")
    print(f"  成功: {success_count} / {len(image_files)}")
    print(f"  失敗: {fail_count} / {len(image_files)}")
    print("=" * 70)


def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description='測試第一階段：人臉檢測')
    parser.add_argument('--mode', type=str, default='image',
                       choices=['image', 'webcam', 'batch'],
                       help='測試模式')
    parser.add_argument('--input', type=str, default=None,
                       help='輸入圖像或目錄路徑')
    
    args = parser.parse_args()
    
    if args.mode == 'webcam':
        test_with_webcam()
    elif args.mode == 'image':
        if args.input is None:
            print("錯誤: 請指定 --input 參數")
            print("範例: python test_stage1.py --mode image --input test_images/face.jpg")
        else:
            test_with_image(args.input)
    elif args.mode == 'batch':
        if args.input is None:
            print("錯誤: 請指定 --input 參數（圖像目錄）")
            print("範例: python test_stage1.py --mode batch --input test_images/")
        else:
            test_batch_images(args.input)


if __name__ == '__main__':
    main()

