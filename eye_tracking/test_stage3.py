"""
測試第三階段：圖像正規化（修正版）
Test Stage 3: Image Normalization (Fixed)

整合第一、二、三階段
"""

import cv2
import numpy as np
from pathlib import Path
import argparse

from stages.stage1_face_detection import FaceDetector
from stages.stage2_head_pose import HeadPoseEstimator
from stages.stage3_normalization import ImageNormalizer
from utils.camera_utils import get_default_camera_matrix


def test_with_image(image_path: str):
    """使用圖像測試階段 1+2+3"""
    print("=" * 70)
    print("測試階段 1+2+3：人臉檢測 + 姿態估計 + 圖像正規化")
    print("=" * 70)
    
    # 讀取圖像
    image = cv2.imread(image_path)
    if image is None:
        print(f"錯誤: 無法讀取圖像 {image_path}")
        return
    
    h, w = image.shape[:2]
    print(f"\n讀取圖像: {image_path}")
    print(f"圖像尺寸: {w} x {h}")
    
    # 生成相機內參矩陣
    camera_matrix = get_default_camera_matrix(w, h)
    
    # ==================== 階段 1：人臉檢測 ====================
    print("\n" + "=" * 70)
    print("[階段 1/3] 人臉檢測與特徵點定位")
    print("=" * 70)
    
    detector = FaceDetector(config={'min_confidence': 0.3})
    face_result = detector.detect(image)
    
    if face_result is None:
        print(" 未檢測到人臉")
        return
    
    print("✓ 檢測成功")
    print(f"  - 特徵點數量: {face_result['num_landmarks']}")
    print(f"  - ETH-XGaze 6 點: {len(face_result['landmarks_2d_selected'])}")
    
    # ==================== 階段 2：頭部姿態估計 ====================
    print("\n" + "=" * 70)
    print("[階段 2/3] 頭部姿態估計（使用 ETH-XGaze 6 點模型）")
    print("=" * 70)
    
    estimator = HeadPoseEstimator(config={
        'face_model_path': 'models/face_model_ethxgaze.txt',  # ETH-XGaze 6 點模型
        'use_iterative': True
    })
    
    # 使用 ETH-XGaze 6 點進行 solvePnP
    pose_result = estimator.estimate(
        landmarks_2d=face_result['landmarks_2d_selected'],  # 使用 6 點
        camera_matrix=camera_matrix
    )
    
    if not pose_result['success']:
        print(" 姿態估計失敗")
        return
    
    print("✓ 姿態估計成功")
    print(f"\n頭部姿態（歐拉角）:")
    print(f"  - Pitch (俯仰): {pose_result['euler_angles']['pitch']:>7.2f}°")
    print(f"  - Yaw   (偏航): {pose_result['euler_angles']['yaw']:>7.2f}°")
    print(f"  - Roll  (翻滾): {pose_result['euler_angles']['roll']:>7.2f}°")
    
    # ==================== 階段 3：圖像正規化 ====================
    print("\n" + "=" * 70)
    print("[階段 3/3] 圖像正規化（使用正確參數）")
    print("=" * 70)
    
    #  修正：使用 ETH-XGaze 6 點模型
    normalizer = ImageNormalizer(config={
        'output_size': (448, 448),
        'focal_norm': 960.0,      # 調整以覆蓋完整臉部（FOV ~19°）      
        'distance_norm': 60.0,    #  正確：60 cm（不是 600mm！）
        'face_model_path': 'models/face_model_ethxgaze.txt'  # ETH-XGaze 6 點模型
    })
    
    print("⚠️  重要：已修正參數")
    print("  - focal_norm: 600 (調整以覆蓋完整臉部)")
    print("  - distance_norm: 60 cm")
    
    norm_result = normalizer.normalize(
        image=image,
        rotation_vector=pose_result['rvec'],
        translation_vector=pose_result['tvec'],
        camera_matrix=camera_matrix
    )
    
    if not norm_result['success']:
        print(" 圖像正規化失敗")
        return
    
    print("✓ 圖像正規化成功")
    print(f"\n正規化結果:")
    print(f"  - 輸出尺寸: {norm_result['normalized_image'].shape}")
    
    # 顯示 scale factor（如果有）
    if 'scale_factor' in norm_result:
        print(f"  - Scale factor: {norm_result['scale_factor']:.3f}")
    
    # 顯示 face center distance（如果有）
    if 'face_center_distance' in norm_result:
        print(f"  - Face center distance: {norm_result['face_center_distance']:.2f} cm")
    
    # ==================== 可視化結果 ====================
    print("\n" + "=" * 70)
    print("生成可視化結果")
    print("=" * 70)
    
    # 1. 原始圖像 + 特徵點 + 坐標軸
    vis_original = detector.visualize_landmarks(
        image.copy(),
        landmarks_selected=face_result['landmarks_2d_selected'],
        bbox=face_result['bbox']
    )
    
    vis_original = estimator.draw_axes(
        vis_original,
        rvec=pose_result['rvec'],
        tvec=pose_result['tvec'],
        camera_matrix=camera_matrix,
        axis_length=100
    )
    
    # 添加姿態信息
    y_offset = 30
    info_texts = [
        f"Pitch: {pose_result['euler_angles']['pitch']:.1f}deg",
        f"Yaw:   {pose_result['euler_angles']['yaw']:.1f}deg",
        f"Roll:  {pose_result['euler_angles']['roll']:.1f}deg",
        f"Distance: {norm_result.get('face_center_distance', 0):.1f}cm",
        f"Scale: {norm_result.get('scale_factor', 0):.3f}"
    ]
    
    for text in info_texts:
        cv2.putText(vis_original, text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 25
    
    # 2. 正規化後的圖像
    normalized_image = norm_result['normalized_image']
    
    # 添加標籤
    cv2.putText(normalized_image, 'Normalized (448x448)', (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # 3. 對比圖
    vis_comparison = normalizer.visualize_normalization(
        image,
        normalized_image,
        norm_result['warp_matrix'],
        norm_result.get('face_center_distance'),
        norm_result.get('scale_factor')
    )
    
    # 保存結果
    output_path = Path('output')
    output_path.mkdir(parents=True, exist_ok=True)
    
    cv2.imwrite(str(output_path / 'test_stage3_original_fixed.jpg'), vis_original)
    cv2.imwrite(str(output_path / 'test_stage3_normalized_fixed.jpg'), normalized_image)
    cv2.imwrite(str(output_path / 'test_stage3_comparison_fixed.jpg'), vis_comparison)
    
    print(f"✓ 結果已保存到 output/")
    print(f"  - test_stage3_original_fixed.jpg")
    print(f"  - test_stage3_normalized_fixed.jpg")
    print(f"  - test_stage3_comparison_fixed.jpg")
    
    # 顯示結果
    cv2.imshow('Original with Landmarks', vis_original)
    cv2.imshow('Normalized Face', normalized_image)
    cv2.imshow('Comparison', vis_comparison)
    
    print("\n按任意鍵關閉...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 70)
    print("測試完成！")
    print("=" * 70)


def test_with_webcam():
    """使用網路攝像頭即時測試"""
    print("=" * 70)
    print("即時測試：人臉檢測 + 姿態估計 + 圖像正規化")
    print("按 'q' 鍵退出")
    print("=" * 70)
    
    # 初始化三個階段（使用正確參數）
    detector = FaceDetector(config={'min_confidence': 0.3})
    estimator = HeadPoseEstimator(config={
        'face_model_path': 'models/face_model_ethxgaze.txt',
        'use_iterative': True
    })
    
    #  修正：使用正確參數
    normalizer = ImageNormalizer(config={
        'output_size': (448, 448),
        'focal_norm': 960.0,      
        'distance_norm': 60.0,    #  正確：60 cm
        'face_model_path': 'models/face_model_ethxgaze.txt'
    })
    
    print("\n⚠️  已使用正確參數:")
    print("  - focal_norm: 960")
    print("  - distance_norm: 60 cm\n")
    
    # 打開攝像頭
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("錯誤: 無法打開攝像頭")
        return
    
    # 獲取攝像頭參數
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    camera_matrix = get_default_camera_matrix(width, height)
    
    print(f"攝像頭已開啟 ({width}x{height})")
    print("開始檢測...\n")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # 階段 1：人臉檢測
        face_result = detector.detect(frame)
        
        if face_result is not None:
            # 階段 2：頭部姿態估計
            pose_result = estimator.estimate(
                landmarks_2d=face_result['landmarks_2d_selected'],
                camera_matrix=camera_matrix
            )
            
            if pose_result['success']:
                # 階段 3：圖像正規化
                norm_result = normalizer.normalize(
                    image=frame,
                    rotation_vector=pose_result['rvec'],
                    translation_vector=pose_result['tvec'],
                    camera_matrix=camera_matrix
                )
                
                # 繪製原始圖像的可視化
                vis_frame = detector.visualize_landmarks(
                    frame,
                    landmarks_selected=face_result['landmarks_2d_selected'],
                    bbox=face_result['bbox']
                )
                
                vis_frame = estimator.draw_axes(
                    vis_frame,
                    rvec=pose_result['rvec'],
                    tvec=pose_result['tvec'],
                    camera_matrix=camera_matrix,
                    axis_length=100
                )
                
                # 顯示姿態和正規化信息
                info_texts = [
                    f"Pitch: {pose_result['euler_angles']['pitch']:>6.1f}deg",
                    f"Yaw:   {pose_result['euler_angles']['yaw']:>6.1f}deg",
                    f"Roll:  {pose_result['euler_angles']['roll']:>6.1f}deg",
                    f"Dist:  {norm_result.get('face_center_distance', 0):>6.1f}cm",
                    f"Scale: {norm_result.get('scale_factor', 0):>6.3f}",
                    f"Frame: {frame_count}"
                ]
                
                y_offset = 30
                for text in info_texts:
                    cv2.putText(vis_frame, text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    y_offset += 22
                
                # 顯示畫面
                cv2.imshow('Original + Detection + Pose', vis_frame)
                
                # 顯示正規化圖像（如果成功）
                if norm_result['success']:
                    normalized_image = norm_result['normalized_image'].copy()
                    
                    # 在正規化圖像上添加信息
                    cv2.putText(normalized_image, 'Normalized', (10, 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(normalized_image, 
                               f"Scale: {norm_result.get('scale_factor', 0):.3f}",
                               (10, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    cv2.imshow('Normalized Face (448x448)', normalized_image)
                
                frame = vis_frame
        else:
            # 未檢測到人臉
            cv2.putText(frame, "No Face Detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow('Original + Detection + Pose', frame)
        
        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n測試結束")


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description='測試階段 1+2+3：人臉檢測 + 姿態估計 + 圖像正規化（修正版）'
    )
    parser.add_argument('--mode', type=str, default='image',
                       choices=['image', 'webcam'],
                       help='測試模式')
    parser.add_argument('--input', type=str, default=None,
                       help='輸入圖像路徑（僅 image 模式）')
    
    args = parser.parse_args()
    
    if args.mode == 'webcam':
        test_with_webcam()
    elif args.mode == 'image':
        if args.input is None:
            print("錯誤: 請指定 --input 參數")
            print("範例: python test_stage3_fixed.py --mode image --input test_images/face.jpg")
        else:
            test_with_image(args.input)


if __name__ == '__main__':
    main()