"""
測試第二階段：頭部姿態估計
Test Stage 2: Head Pose Estimation

整合第一階段和第二階段
"""

import cv2
import numpy as np
from pathlib import Path
import argparse

from stages.stage1_face_detection import FaceDetector
from stages.stage2_head_pose import HeadPoseEstimator
from utils.camera_utils import get_default_camera_matrix


def test_with_image(image_path: str):
    image = cv2.imread(image_path)
    if image is None: return

    # --- 關鍵修正：將超大圖像縮放至合理尺寸 ---
    max_side = 1280
    h, w = image.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
        h, w = image.shape[:2]
        print(f"⚠️ 已將圖像縮放至: {w} x {h} 以提高 PnP 穩定性")
    # ---------------------------------------

    camera_matrix = get_default_camera_matrix(w, h)
    
    # ==================== 階段 1：人臉檢測 ====================
    print("\n" + "=" * 70)
    print("[階段 1/2] 人臉檢測與特徵點定位")
    print("=" * 70)
    
    detector = FaceDetector(config={'min_confidence': 0.3})
    face_result = detector.detect(image)
    
    if face_result is None:
        print(" 未檢測到人臉")
        return
    
    print("✓ 檢測成功")
    print(f"  - 特徵點數量: {face_result['num_landmarks']}")
    print(f"  - 關鍵點數量: {len(face_result['landmarks_2d_selected'])}")
    
    # ==================== 階段 2：頭部姿態估計 ====================
    print("\n" + "=" * 70)
    print("[階段 2/2] 頭部姿態估計")
    print("=" * 70)
    
    estimator = HeadPoseEstimator(config={
        'face_model_path': 'models/face_model_ethxgaze.txt'
    })
    
    pose_result = estimator.estimate(
        landmarks_2d=face_result['landmarks_2d_selected'],
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
    
    # 獲取頭部朝向
    direction = estimator.get_head_direction_vector(pose_result['rotation_matrix'])
    print(f"\n頭部朝向向量:")
    print(f"  - X: {direction[0]:>7.4f}")
    print(f"  - Y: {direction[1]:>7.4f}")
    print(f"  - Z: {direction[2]:>7.4f}")
    
    # ==================== 可視化結果 ====================
    print("\n" + "=" * 70)
    print("生成可視化結果")
    print("=" * 70)
    
    # 繪製特徵點
    vis_image = detector.visualize_landmarks(
        image.copy(),
        landmarks_selected=face_result['landmarks_2d_selected'],
        bbox=face_result['bbox']
    )
    
    # 繪製 3D 坐標軸
    vis_image = estimator.draw_axes(
        vis_image,
        rvec=pose_result['rvec'],
        tvec=pose_result['tvec'],
        camera_matrix=camera_matrix,
        axis_length=100
    )
    
    # 添加文字資訊
    y_offset = 30
    info_texts = [
        f"Pitch: {pose_result['euler_angles']['pitch']:.1f}deg",
        f"Yaw:   {pose_result['euler_angles']['yaw']:.1f}deg",
        f"Roll:  {pose_result['euler_angles']['roll']:.1f}deg"
    ]
    
    for text in info_texts:
        cv2.putText(vis_image, text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 30
    
    # 保存結果
    output_path = Path('output/test_stage2_result.jpg')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis_image)
    print(f"✓ 結果已保存: {output_path}")
    
    # 顯示結果
    cv2.imshow('Stage 1+2 Result', vis_image)
    print("\n按任意鍵關閉...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 70)
    print("測試完成！")
    print("=" * 70)


def test_with_webcam():
    """使用網路攝像頭即時測試"""
    print("=" * 70)
    print("即時測試：人臉檢測 + 頭部姿態估計")
    print("按 'q' 鍵退出")
    print("=" * 70)
    
    # 初始化
    detector = FaceDetector(config={'min_confidence': 0.5})
    estimator = HeadPoseEstimator(config={
        'face_model_path': 'models/face_model_ethxgaze.txt'
    })
    
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
                # 繪製特徵點
                vis_frame = detector.visualize_landmarks(
                    frame,
                    landmarks_selected=face_result['landmarks_2d_selected'],
                    bbox=face_result['bbox']
                )
                
                # 繪製 3D 坐標軸
                vis_frame = estimator.draw_axes(
                    vis_frame,
                    rvec=pose_result['rvec'],
                    tvec=pose_result['tvec'],
                    camera_matrix=camera_matrix,
                    axis_length=100
                )
                
                # 顯示資訊
                info_texts = [
                    f"Pitch: {pose_result['euler_angles']['pitch']:>6.1f}deg",
                    f"Yaw:   {pose_result['euler_angles']['yaw']:>6.1f}deg",
                    f"Roll:  {pose_result['euler_angles']['roll']:>6.1f}deg",
                    f"Frame: {frame_count}"
                ]
                
                y_offset = 30
                for text in info_texts:
                    cv2.putText(vis_frame, text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y_offset += 25
                
                frame = vis_frame
        else:
            # 未檢測到人臉
            cv2.putText(frame, "No Face Detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 顯示畫面
        cv2.imshow('Head Pose Estimation', frame)
        
        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n測試結束")


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description='測試階段 1+2：人臉檢測與頭部姿態估計'
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
            print("範例: python test_stage2.py --mode image --input test_images/face.jpg")
        else:
            test_with_image(args.input)


if __name__ == '__main__':
    main()

