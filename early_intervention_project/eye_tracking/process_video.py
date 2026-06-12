"""
影片視線估計處理腳本
Video Gaze Estimation Processor

功能：
- 讀取影片文件
- 逐幀進行視線估計
- 輸出帶視線標註的影片
- 導出每一幀的視線數據（CSV 格式）
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import time
from tqdm import tqdm

# 導入視線估計系統
from stages.stage1_face_detection import FaceDetector
from stages.stage2_head_pose import HeadPoseEstimator
from stages.stage3_normalization import ImageNormalizer
from stages.stage4_gaze_network import GazeEstimator
from stages.stage5_gaze_vector import GazeVectorConverter

from utils.camera_utils import get_default_camera_matrix
from utils.visualization import draw_gaze_with_face_box


class VideoGazeProcessor:
    """影片視線估計處理器"""
    
    def __init__(self):
        """初始化影片處理器"""
        print("=" * 70)
        print("視線估計影片處理器初始化")
        print("Video Gaze Estimation Processor Initialization")
        print("=" * 70)
        
        # 初始化各個階段
        print("\n正在載入模型...")
        
        self.face_detector = FaceDetector(config={'min_confidence': 0.3})
        
        self.head_pose_estimator = HeadPoseEstimator(config={
            'face_model_path': 'models/face_model_ethxgaze.txt',
            'use_iterative': True
        })
        
        self.image_normalizer = ImageNormalizer(config={
            'output_size': (224, 224),
            'focal_norm': 960.0,
            'distance_norm': 60.0,
            'face_model_path': 'models/face_model_ethxgaze.txt'
        })
        
        self.gaze_estimator = GazeEstimator(config={
            'model_path': 'models/epoch_24_ckpt.pth.tar',
            'use_gpu': False
        })
        
        self.gaze_converter = GazeVectorConverter()
        
        print("✓ 模型載入完成！\n")
    
    def process_video(self, 
                     video_path, 
                     output_video_path=None, 
                     output_csv_path=None,
                     show_preview=False, 
                     skip_frames=0, 
                     max_frames=None):
        """
        處理影片文件
        
        Args:
            video_path: 輸入影片路徑
            output_video_path: 輸出影片路徑（如果為 None，不輸出影片）
            output_csv_path: 輸出 CSV 路徑（如果為 None，不輸出 CSV）
            show_preview: 是否顯示處理預覽
            skip_frames: 跳過的幀數（例如：skip_frames=2 表示每3幀處理1幀）
            max_frames: 最大處理幀數（用於測試）
        
        Returns:
            results_df: 包含所有幀視線數據的 DataFrame
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"影片文件不存在: {video_path}")
        
        print(f"\n處理影片: {video_path}")
        print("=" * 70)
        
        # 打開影片
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"無法打開影片: {video_path}")
        
        # 獲取影片資訊
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"影片資訊:")
        print(f"  - 解析度: {width} x {height}")
        print(f"  - FPS: {fps}")
        print(f"  - 總幀數: {total_frames}")
        print(f"  - 時長: {total_frames/fps:.2f} 秒")
        
        if skip_frames > 0:
            print(f"  - 跳幀設定: 每 {skip_frames + 1} 幀處理 1 幀")
        if max_frames:
            print(f"  - 最大處理幀數: {max_frames}")
        
        # 生成相機內參矩陣
        camera_matrix = get_default_camera_matrix(width, height)
        
        # 初始化影片寫入器（如果需要）
        writer = None
        if output_video_path:
            output_video_path = Path(output_video_path)
            output_video_path.parent.mkdir(parents=True, exist_ok=True)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                str(output_video_path),
                fourcc,
                fps,
                (width, height)
            )
            print(f"\n輸出影片: {output_video_path}")
        
        # 初始化數據收集
        results_data = []
        
        # 處理統計
        frame_idx = 0
        processed_count = 0
        failed_count = 0
        start_time = time.time()
        
        # 進度條
        pbar = tqdm(total=total_frames, desc="處理進度", unit="幀")
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # 更新進度條
                pbar.update(1)
                
                # 檢查是否達到最大幀數限制
                if max_frames and processed_count >= max_frames:
                    print(f"\n已達到最大處理幀數限制: {max_frames}")
                    break
                
                # 跳幀處理
                if frame_idx % (skip_frames + 1) != 0:
                    frame_idx += 1
                    
                    # 如果需要輸出影片，寫入原始幀
                    if writer:
                        writer.write(frame)
                    
                    continue
                
                # 處理當前幀
                try:
                    result = self._process_frame(frame, frame_idx, camera_matrix, fps)
                    
                    if result is not None:
                        results_data.append(result)
                        processed_count += 1
                        
                        # 繪製視線方向（使用 test_gaze_arrow.py 的風格）
                        annotated_frame = self._annotate_frame(frame.copy(), result)
                        
                        # 顯示預覽
                        if show_preview:
                            cv2.imshow('Gaze Estimation', annotated_frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                print("\n用戶中斷處理")
                                break
                        
                        # 寫入輸出影片
                        if writer:
                            writer.write(annotated_frame)
                    else:
                        failed_count += 1
                        
                        # 寫入原始幀（未檢測到人臉）
                        if writer:
                            writer.write(frame)
                
                except Exception as e:
                    # 處理失敗，記錄但繼續
                    failed_count += 1
                    if writer:
                        writer.write(frame)
                
                frame_idx += 1
        
        finally:
            # 清理資源
            pbar.close()
            cap.release()
            if writer:
                writer.release()
            if show_preview:
                cv2.destroyAllWindows()
        
        # 處理統計
        elapsed_time = time.time() - start_time
        
        print("\n" + "=" * 70)
        print("處理完成！")
        print("=" * 70)
        print(f"總幀數: {frame_idx}")
        print(f"成功處理: {processed_count} 幀")
        print(f"失敗/跳過: {failed_count} 幀")
        if processed_count + failed_count > 0:
            print(f"成功率: {processed_count/(processed_count+failed_count)*100:.1f}%")
        print(f"總耗時: {elapsed_time:.2f} 秒")
        print(f"平均速度: {frame_idx/elapsed_time:.2f} FPS")
        
        # 轉換為 DataFrame
        if results_data:
            results_df = pd.DataFrame(results_data)
            
            # 保存 CSV（如果需要）
            if output_csv_path:
                output_csv_path = Path(output_csv_path)
                output_csv_path.parent.mkdir(parents=True, exist_ok=True)
                results_df.to_csv(output_csv_path, index=False)
                print(f"\n✓ 視線數據已保存到: {output_csv_path}")
            
            # 顯示統計摘要
            print("\n" + "=" * 70)
            print("視線估計統計摘要:")
            print("=" * 70)
            print(f"Pitch (俯仰角):")
            print(f"  - 平均值: {results_df['gaze_pitch_deg'].mean():.2f}°")
            print(f"  - 標準差: {results_df['gaze_pitch_deg'].std():.2f}°")
            print(f"  - 範圍: [{results_df['gaze_pitch_deg'].min():.2f}°, {results_df['gaze_pitch_deg'].max():.2f}°]")
            
            print(f"\nYaw (偏航角):")
            print(f"  - 平均值: {results_df['gaze_yaw_deg'].mean():.2f}°")
            print(f"  - 標準差: {results_df['gaze_yaw_deg'].std():.2f}°")
            print(f"  - 範圍: [{results_df['gaze_yaw_deg'].min():.2f}°, {results_df['gaze_yaw_deg'].max():.2f}°]")
            
            print("=" * 70 + "\n")
            
            return results_df
        else:
            print("\n警告: 沒有成功處理任何幀！")
            return None
    
    def _process_frame(self, frame, frame_idx, camera_matrix, fps):
        """
        處理單個幀
        
        Args:
            frame: 輸入幀（BGR）
            frame_idx: 幀索引
            camera_matrix: 相機內參矩陣
            fps: 影片幀率
        
        Returns:
            result: 包含視線數據的字典，如果失敗返回 None
        """
        try:
            # 第一階段：人臉檢測
            face_result = self.face_detector.detect(frame)
            
            if face_result is None:
                return None
            
            # 第二階段：頭部姿態估計
            pose_result = self.head_pose_estimator.estimate(
                landmarks_2d=face_result['landmarks_2d_selected'],
                camera_matrix=camera_matrix
            )
            
            if not pose_result['success']:
                return None
            
            # 第三階段：圖像正規化
            norm_result = self.image_normalizer.normalize(
                image=frame,
                rotation_vector=pose_result['rvec'],
                translation_vector=pose_result['tvec'],
                camera_matrix=camera_matrix
            )
            
            if not norm_result['success']:
                return None
            
            # 第四階段：神經網絡推理
            gaze_result = self.gaze_estimator.estimate(norm_result['normalized_image'])
            
            if not gaze_result['success']:
                return None
            
            # 第五階段：視線向量轉換
            gaze_vector = self.gaze_converter.angles_to_vector(
                pitch=gaze_result['gaze_angles'][0],
                yaw=gaze_result['gaze_angles'][1]
            )
            
            # 計算眼睛位置
            landmarks = face_result['landmarks_2d_selected']
            left_eye = tuple(((landmarks[0] + landmarks[1]) / 2).astype(int))
            right_eye = tuple(((landmarks[2] + landmarks[3]) / 2).astype(int))
            nose_tip = tuple(landmarks[4].astype(int))
            
            # 組裝結果
            result = {
                'frame_idx': frame_idx,
                'timestamp_sec': frame_idx / fps,
                
                # 頭部姿態（度）
                'head_pitch_deg': pose_result['euler_angles']['pitch'],
                'head_yaw_deg': pose_result['euler_angles']['yaw'],
                'head_roll_deg': pose_result['euler_angles']['roll'],
                
                # 視線角度（弧度）
                'gaze_pitch_rad': gaze_result['gaze_angles'][0],
                'gaze_yaw_rad': gaze_result['gaze_angles'][1],
                
                # 視線角度（度）
                'gaze_pitch_deg': gaze_result['gaze_angles_deg'][0],
                'gaze_yaw_deg': gaze_result['gaze_angles_deg'][1],
                
                # 3D 視線向量
                'gaze_vector_x': gaze_vector[0],
                'gaze_vector_y': gaze_vector[1],
                'gaze_vector_z': gaze_vector[2],
                
                # 人臉位置
                'face_bbox_x': face_result['bbox'][0],
                'face_bbox_y': face_result['bbox'][1],
                'face_bbox_w': face_result['bbox'][2] - face_result['bbox'][0],
                'face_bbox_h': face_result['bbox'][3] - face_result['bbox'][1],
                
                # 保存用於繪製的數據
                '_face_bbox': face_result['bbox'],
                '_pitch': gaze_result['gaze_angles'][0],
                '_yaw': gaze_result['gaze_angles'][1],
                '_gaze_vector': gaze_vector,
                '_nose_tip': nose_tip,
                '_left_eye': left_eye,
                '_right_eye': right_eye,
            }
            
            return result
        
        except Exception as e:
            # 靜默失敗，返回 None
            return None
    
    def _annotate_frame(self, frame, result):
        """
        在幀上繪製視線方向和資訊（使用 test_gaze_arrow.py 的風格）
        
        Args:
            frame: 輸入幀
            result: 處理結果
        
        Returns:
            annotated_frame: 標註後的幀
        """
        # 使用與 test_gaze_arrow.py 相同的可視化函數
        frame = draw_gaze_with_face_box(
            frame,
            face_bbox=result['_face_bbox'],
            pitch=result['_pitch'],
            yaw=result['_yaw'],
            gaze_vector=result['_gaze_vector'],
            nose_tip=result['_nose_tip'],
            left_eye=result['_left_eye'],
            right_eye=result['_right_eye'],
            confidence=result.get('_confidence'),  #  新增置信度
            show_angles=True,
            show_direction_label=False,  # 關閉方向標籤（Center, Looking Up 等）
            show_gaze_vector=True,  #  顯示 3D 向量（可選）
            bbox_format='xyxy'  #  關鍵修復！
        )
        
        return frame


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description='視線估計影片處理器 - Video Gaze Estimation Processor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 基本使用（處理影片並導出數據）
  python process_video.py --input video.mp4 --output output.mp4 --csv data.csv
  
  # 僅導出數據（不輸出影片，速度更快）
  python process_video.py --input video.mp4 --csv data.csv
  
  # 跳幀處理（速度提升 3 倍）
  python process_video.py --input video.mp4 --skip-frames 2 --csv data.csv
  
  # 測試模式（只處理前 100 幀）
  python process_video.py --input video.mp4 --max-frames 100 --csv test.csv
  
  # 顯示即時預覽
  python process_video.py --input video.mp4 --output output.mp4 --show-preview
        """
    )
    
    parser.add_argument('--input', type=str, required=True,
                       help='輸入影片路徑')
    parser.add_argument('--output', type=str, default=None,
                       help='輸出影片路徑（可選，不指定則不輸出影片）')
    parser.add_argument('--csv', type=str, default=None,
                       help='輸出 CSV 數據路徑（可選）')
    parser.add_argument('--show-preview', action='store_true',
                       help='顯示處理預覽窗口')
    parser.add_argument('--skip-frames', type=int, default=0,
                       help='跳幀處理（0=處理所有幀，1=每2幀處理1幀，2=每3幀處理1幀）')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='最大處理幀數（用於測試）')
    
    args = parser.parse_args()
    
    # 檢查輸入文件
    if not Path(args.input).exists():
        print(f"錯誤: 輸入影片不存在: {args.input}")
        return 1
    
    # 檢查是否至少指定了一個輸出
    if args.output is None and args.csv is None:
        print("警告: 未指定輸出影片或 CSV 文件")
        print("建議至少指定 --output 或 --csv 其中一個")
        response = input("是否繼續？(y/n): ")
        if response.lower() != 'y':
            return 0
    
    try:
        # 初始化處理器
        processor = VideoGazeProcessor()
        
        # 處理影片
        results_df = processor.process_video(
            video_path=args.input,
            output_video_path=args.output,
            output_csv_path=args.csv,
            show_preview=args.show_preview,
            skip_frames=args.skip_frames,
            max_frames=args.max_frames
        )
        
        if results_df is not None:
            print("\n✓ 處理完成！")
            return 0
        else:
            print("\n⚠ 處理完成，但沒有成功處理任何幀")
            return 1
        
    except KeyboardInterrupt:
        print("\n\n用戶中斷處理")
        return 1
    except Exception as e:
        print(f"\n錯誤: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
