"""
測試第四階段：神經網絡推理
Test Stage 4: Neural Network Inference

整合第一、二、三、四階段
"""

import cv2
import numpy as np
from pathlib import Path
import argparse

from stages.stage1_face_detection import FaceDetector
from stages.stage2_head_pose import HeadPoseEstimator
from stages.stage3_normalization import ImageNormalizer
from stages.stage4_gaze_network import GazeEstimator
from utils.camera_utils import get_default_camera_matrix


def test_with_image(image_path: str):
    """使用圖像測試階段 1+2+3+4"""
    print("=" * 70)
    print("測試階段 1+2+3+4：完整視線估計流程")
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
    print("[階段 1/4] 人臉檢測與特徵點定位")
    print("=" * 70)
    
    detector = FaceDetector(config={'min_confidence': 0.3})
    face_result = detector.detect(image)
    
    if face_result is None:
        print(" 未檢測到人臉")
        return
    
    print("✓ 檢測成功")
    
    # ==================== 階段 2：頭部姿態估計 ====================
    print("\n" + "=" * 70)
    print("[階段 2/4] 頭部姿態估計")
    print("=" * 70)
    
    estimator = HeadPoseEstimator(config={
        'face_model_path': 'models/face_model_ethxgaze.txt',
        'use_iterative': True
    })
    
    pose_result = estimator.estimate(
        landmarks_2d=face_result['landmarks_2d_selected'],
        camera_matrix=camera_matrix
    )
    
    if not pose_result['success']:
        print(" 姿態估計失敗")
        return
    
    print("✓ 姿態估計成功")
    print(f"  Pitch: {pose_result['euler_angles']['pitch']:.1f}°, "
          f"Yaw: {pose_result['euler_angles']['yaw']:.1f}°, "
          f"Roll: {pose_result['euler_angles']['roll']:.1f}°")
    
    # ==================== 階段 3：圖像正規化 ====================
    print("\n" + "=" * 70)
    print("[階段 3/4] 圖像正規化")
    print("=" * 70)
    
    normalizer = ImageNormalizer(config={
        'output_size': (224, 224),
        'focal_norm': 960.0,
        'distance_norm': 60.0,
        'face_model_path': 'models/face_model_ethxgaze.txt'
    })
    
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
    print(f"  輸出尺寸: {norm_result['normalized_image'].shape}")
    
    # ==================== 階段 4：神經網絡推理 ====================
    print("\n" + "=" * 70)
    print("[階段 4/4] 神經網絡推理")
    print("=" * 70)
    
    try:
        gaze_estimator = GazeEstimator(config={
            'model_path': 'models/epoch_24_ckpt.pth.tar',
            'use_gpu': True
        })
    except Exception as e:
        print(f" 無法載入視線估計模型: {e}")
        return
    
    gaze_result = gaze_estimator.estimate(norm_result['normalized_image'])
    
    if not gaze_result['success']:
        print(" 視線估計失敗")
        return
    
    print("✓ 視線估計成功")
    print(f"\n視線方向:")
    print(f"  Pitch: {gaze_result['gaze_angles_deg'][0]:>7.2f}°")
    print(f"  Yaw:   {gaze_result['gaze_angles_deg'][1]:>7.2f}°")
    
    # ==================== 可視化結果 ====================
    print("\n" + "=" * 70)
    print("生成可視化結果")
    print("=" * 70)
    
    # 1. 原始圖像 + 特徵點 + 頭部坐標軸
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
    
    # 添加頭部姿態信息
    y_offset = 30
    info_texts = [
        f"Head Pose:",
        f"  Pitch: {pose_result['euler_angles']['pitch']:>6.1f}deg",
        f"  Yaw:   {pose_result['euler_angles']['yaw']:>6.1f}deg",
        f"  Roll:  {pose_result['euler_angles']['roll']:>6.1f}deg",
    ]
    
    for text in info_texts:
        cv2.putText(vis_original, text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 25
    
    # 2. 正規化圖像 + 視線方向
    vis_normalized = gaze_estimator.draw_gaze_on_image(
        norm_result['normalized_image'].copy(),
        pitch=gaze_result['gaze_angles'][0],
        yaw=gaze_result['gaze_angles'][1],
        thickness=3
    )
    
    # 添加視線信息
    cv2.putText(vis_normalized, 'Normalized + Gaze', (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(vis_normalized, 
               f"Gaze Pitch: {gaze_result['gaze_angles_deg'][0]:.1f}deg", 
               (10, 200),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(vis_normalized, 
               f"Gaze Yaw: {gaze_result['gaze_angles_deg'][1]:.1f}deg", 
               (10, 220),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 保存結果
    output_path = Path('output')
    output_path.mkdir(parents=True, exist_ok=True)
    
    cv2.imwrite(str(output_path / 'test_stage4_original.jpg'), vis_original)
    cv2.imwrite(str(output_path / 'test_stage4_normalized_gaze.jpg'), vis_normalized)
    
    print(f"✓ 結果已保存到 output/")
    print(f"  - test_stage4_original.jpg")
    print(f"  - test_stage4_normalized_gaze.jpg")
    
    # 顯示結果
    cv2.imshow('Original + Detection + Pose', vis_original)
    cv2.imshow('Normalized + Gaze Direction', vis_normalized)
    
    print("\n按任意鍵關閉...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 70)
    print("✅ 完整流程測試完成！")
    print("=" * 70)


def test_with_webcam():
    """使用網路攝像頭即時測試完整流程"""
    print("=" * 70)
    print("即時測試：完整視線估計流程（四階段）")
    print("按 'q' 鍵退出")
    print("=" * 70)
    
    # 初始化四個階段
    print("\n初始化各階段...")
    detector = FaceDetector(config={'min_confidence': 0.5})
    estimator = HeadPoseEstimator(config={
        'face_model_path': 'models/face_model_ethxgaze.txt',
        'use_iterative': True
    })
    normalizer = ImageNormalizer(config={
        'output_size': (224, 224),
        'focal_norm': 960.0,
        'distance_norm': 60.0,
        'face_model_path': 'models/face_model_ethxgaze.txt'
    })
    
    try:
        gaze_estimator = GazeEstimator(config={
            'model_path': 'models/epoch_24_ckpt.pth.tar',
            'use_gpu': True
        })
    except Exception as e:
        print(f"\n 無法載入視線估計模型: {e}")
        return
    
    # 打開攝像頭
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("錯誤: 無法打開攝像頭")
        return
    
    # 獲取攝像頭參數
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    camera_matrix = get_default_camera_matrix(width, height)
    
    print(f"\n攝像頭已開啟 ({width}x{height})")
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
                
                if norm_result['success']:
                    # 階段 4：視線估計
                    gaze_result = gaze_estimator.estimate(norm_result['normalized_image'])
                    
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
                    
                    # 顯示信息
                    info_texts = [
                        f"Head: P={pose_result['euler_angles']['pitch']:>5.1f} "
                        f"Y={pose_result['euler_angles']['yaw']:>5.1f} "
                        f"R={pose_result['euler_angles']['roll']:>5.1f}",
                    ]
                    
                    if gaze_result['success']:
                        info_texts.append(
                            f"Gaze: P={gaze_result['gaze_angles_deg'][0]:>5.1f} "
                            f"Y={gaze_result['gaze_angles_deg'][1]:>5.1f}"
                        )
                    
                    info_texts.append(f"FPS: {frame_count}")
                    
                    y_offset = 30
                    for text in info_texts:
                        cv2.putText(vis_frame, text, (10, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        y_offset += 25
                    
                    frame = vis_frame
                    
                    # 顯示正規化圖像 + 視線
                    if gaze_result['success']:
                        vis_normalized = gaze_estimator.draw_gaze_on_image(
                            norm_result['normalized_image'],
                            pitch=gaze_result['gaze_angles'][0],
                            yaw=gaze_result['gaze_angles'][1],
                        )
                        cv2.imshow('Normalized + Gaze', vis_normalized)
        else:
            # 未檢測到人臉
            cv2.putText(frame, "No Face Detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 顯示主畫面
        cv2.imshow('Full Pipeline', frame)
        
        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n測試結束")


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description='測試階段 1+2+3+4：完整視線估計流程'
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
            print("範例: python test_stage4.py --mode image --input test_images/face.jpg")
        else:
            test_with_image(args.input)


if __name__ == '__main__':
    main()

