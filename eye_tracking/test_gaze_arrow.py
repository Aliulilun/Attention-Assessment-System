"""
測試視線箭頭可視化
Test Gaze Arrow Visualization

展示像附圖那樣的視線方向箭頭
"""

import cv2
import numpy as np
from pathlib import Path
import argparse

from stages.stage1_face_detection import FaceDetector
from stages.stage2_head_pose import HeadPoseEstimator
from stages.stage3_normalization import ImageNormalizer
from stages.stage4_gaze_network import GazeEstimator
from stages.stage5_gaze_vector import GazeVectorConverter
from utils.camera_utils import get_default_camera_matrix
from utils.visualization import draw_gaze_with_face_box, draw_multiple_gazes


def test_gaze_arrow_with_image(image_path: str):
    """使用圖像測試視線箭頭顯示"""
    print("=" * 80)
    print("視線箭頭可視化測試")
    print("=" * 80)
    
    # 讀取圖像
    image = cv2.imread(image_path)
    if image is None:
        print(f"錯誤: 無法讀取圖像 {image_path}")
        return
    
    h, w = image.shape[:2]
    print(f"\n讀取圖像: {image_path}")
    print(f"圖像尺寸: {w} x {h}")
    
    camera_matrix = get_default_camera_matrix(w, h)
    
    # 初始化所有階段
    print("\n初始化...")
    detector = FaceDetector(config={'min_confidence': 0.3})
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
        print(f" 無法載入視線估計模型: {e}")
        return
    
    converter = GazeVectorConverter()
    
    # 處理圖像
    print("\n處理圖像...")
    face_result = detector.detect(image)
    
    if face_result is None:
        print(" 未檢測到人臉")
        return
    
    print("✓ 檢測到人臉")
    
    # 姿態估計
    pose_result = estimator.estimate(
        landmarks_2d=face_result['landmarks_2d_selected'],
        camera_matrix=camera_matrix
    )
    
    if not pose_result['success']:
        print(" 姿態估計失敗")
        return
    
    print("✓ 姿態估計成功")
    
    # 圖像正規化
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
    
    # 視線估計
    gaze_result = gaze_estimator.estimate(norm_result['normalized_image'])
    
    if not gaze_result['success']:
        print(" 視線估計失敗")
        return
    
    print("✓ 視線估計成功")
    
    # 視線向量轉換
    gaze_vector = converter.angles_to_vector(
        pitch=gaze_result['gaze_angles'][0],
        yaw=gaze_result['gaze_angles'][1]
    )
    
    print("✓ 視線向量轉換成功")
    
    # 顯示結果
    print(f"\n視線結果:")
    print(f"  Pitch: {gaze_result['gaze_angles_deg'][0]:.1f}°")
    print(f"  Yaw:   {gaze_result['gaze_angles_deg'][1]:.1f}°")
    print(f"  向量:  [{gaze_vector[0]:.3f}, {gaze_vector[1]:.3f}, {gaze_vector[2]:.3f}]")
    
    direction = converter.get_gaze_direction_label(
        gaze_result['gaze_angles_deg'][0],
        gaze_result['gaze_angles_deg'][1]
    )
    print(f"  方向:  {direction}")
    
    # ==================== 可視化（像附圖那樣）====================
    print("\n" + "=" * 80)
    print("生成視線箭頭可視化")
    print("=" * 80)
    
    # 獲取眼睛位置作為箭頭起點
    left_eye = None
    right_eye = None
    nose_tip = None
    
    if 'landmarks_2d_selected' in face_result and face_result['landmarks_2d_selected'] is not None:
        landmarks = face_result['landmarks_2d_selected']
        # 根據 stage1 的 KEY_LANDMARKS_INDICES 順序：
        # 0: left_eye_outer, 1: left_eye_inner, 2: right_eye_inner, 3: right_eye_outer
        # 4: nose_tip, 5: nose_bottom, 6: left_mouth, 7: right_mouth
        
        # 計算左眼中心（外角和內角的中點）
        left_eye_outer = landmarks[0]
        left_eye_inner = landmarks[1]
        left_eye = tuple((left_eye_outer + left_eye_inner) / 2)
        
        # 計算右眼中心（內角和外角的中點）
        right_eye_inner = landmarks[2]
        right_eye_outer = landmarks[3]
        right_eye = tuple((right_eye_inner + right_eye_outer) / 2)
        
        # 鼻尖
        nose_tip = tuple(landmarks[4])
    
    # 使用新的可視化函數
    vis_image = draw_gaze_with_face_box(
        image,
        face_bbox=face_result['bbox'],
        pitch=gaze_result['gaze_angles'][0],
        yaw=gaze_result['gaze_angles'][1],
        gaze_vector=gaze_vector,
        nose_tip=nose_tip,
        left_eye=left_eye,
        right_eye=right_eye,
        show_angles=True,
        show_direction_label=False,  # 關閉方向標籤（Center, Looking Up 等）
        show_gaze_vector=True,  # 顯示 3D 向量
        bbox_format='xyxy'  # stage1 返回的是 [x_min, y_min, x_max, y_max] 格式
    )
    
    # 保存結果
    output_path = Path('output')
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / 'gaze_arrow_visualization.jpg'
    cv2.imwrite(str(output_file), vis_image)
    print(f"✓ 結果已保存: {output_file}")
    
    # 顯示
    cv2.imshow('Gaze Arrow Visualization', vis_image)
    print("\n按任意鍵關閉...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n✅ 完成！")


def test_gaze_arrow_with_webcam():
    """使用網路攝像頭即時顯示視線箭頭"""
    print("=" * 80)
    print("即時視線箭頭可視化")
    print("按 'q' 退出，'s' 截圖")
    print("=" * 80)
    
    # 初始化
    print("\n初始化...")
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
        print(f" 無法載入視線估計模型: {e}")
        return
    
    converter = GazeVectorConverter()
    
    # 打開攝像頭
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("錯誤: 無法打開攝像頭")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    camera_matrix = get_default_camera_matrix(width, height)
    
    print(f"攝像頭已開啟 ({width}x{height})")
    print("開始處理...\n")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # 完整流程
        face_result = detector.detect(frame)
        
        if face_result is not None:
            pose_result = estimator.estimate(
                landmarks_2d=face_result['landmarks_2d_selected'],
                camera_matrix=camera_matrix
            )
            
            if pose_result['success']:
                norm_result = normalizer.normalize(
                    image=frame,
                    rotation_vector=pose_result['rvec'],
                    translation_vector=pose_result['tvec'],
                    camera_matrix=camera_matrix
                )
                
                if norm_result['success']:
                    gaze_result = gaze_estimator.estimate(norm_result['normalized_image'])
                    
                    if gaze_result['success']:
                        # 轉換為向量
                        gaze_vector = converter.angles_to_vector(
                            pitch=gaze_result['gaze_angles'][0],
                            yaw=gaze_result['gaze_angles'][1]
                        )
                        
                        # 獲取眼睛位置作為箭頭起點
                        left_eye = None
                        right_eye = None
                        nose_tip = None
                        
                        if 'landmarks_2d_selected' in face_result and face_result['landmarks_2d_selected'] is not None:
                            landmarks = face_result['landmarks_2d_selected']
                            # 計算左眼中心
                            left_eye = tuple((landmarks[0] + landmarks[1]) / 2)
                            # 計算右眼中心
                            right_eye = tuple((landmarks[2] + landmarks[3]) / 2)
                            # 鼻尖
                            nose_tip = tuple(landmarks[4])
                        
                        # 使用新的可視化函數（像附圖那樣）
                        frame = draw_gaze_with_face_box(
                            frame,
                            face_bbox=face_result['bbox'],
                            pitch=gaze_result['gaze_angles'][0],
                            yaw=gaze_result['gaze_angles'][1],
                            gaze_vector=gaze_vector,
                            nose_tip=nose_tip,
                            left_eye=left_eye,
                            right_eye=right_eye,
                            confidence=face_result['confidence'],
                            show_angles=True,
                            show_direction_label=False,  # 關閉方向標籤
                            show_gaze_vector=True,
                            bbox_format='xyxy'
                        )
        else:
            # 未檢測到人臉
            cv2.putText(frame, "No Face Detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 顯示畫面
        cv2.imshow('Gaze Arrow (Press q to quit, s to save)', frame)
        
        # 按鍵控制
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # 截圖
            output_path = Path('output')
            output_path.mkdir(exist_ok=True)
            filename = f'gaze_arrow_{frame_count}.jpg'
            cv2.imwrite(str(output_path / filename), frame)
            print(f"截圖已保存: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n測試結束")


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description='視線箭頭可視化測試'
    )
    parser.add_argument('--mode', type=str, default='webcam',
                       choices=['image', 'webcam'],
                       help='測試模式')
    parser.add_argument('--input', type=str, default=None,
                       help='輸入圖像路徑（僅 image 模式）')
    
    args = parser.parse_args()
    
    if args.mode == 'webcam':
        test_gaze_arrow_with_webcam()
    elif args.mode == 'image':
        if args.input is None:
            print("錯誤: 請指定 --input 參數")
            print("範例: python test_gaze_arrow.py --mode image --input test_images/face.jpg")
        else:
            test_gaze_arrow_with_image(args.input)


if __name__ == '__main__':
    main()

