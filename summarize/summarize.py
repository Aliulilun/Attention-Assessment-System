import os
import re
import argparse
from pathlib import Path
import pandas as pd

def parse_gaze_log(file_path):
    """
    解析單一 txt 檔案的視線追蹤紀錄。
    涵蓋 Stage 1~10，包含：
    1. 特徵擷取順序調整：將反應等級移至該 Stage 欄位的最後面。
    2. 次級事件回溯 (Fallback Mechanism) 處理 TH#2。
    """
    vid = file_path.stem
    text = file_path.read_text(encoding='utf-8')
        
    vid_match = re.search(r'Video:.*[/\\](\d+)\.mp4', text)
    if vid_match:
        vid = vid_match.group(1)

    parts = text.split('=== Full Event Log ===')
    summary_text = parts[0]
    event_log_text = parts[1] if len(parts) > 1 else ""

    # 定義 Stage Schema (將反應等級移至陣列最尾端)
    stages_schema = {
        1:  ['T0', 'TB', 'TH', 'Pointing次數', 'TB次數', 'TH次數', '反應等級'],
        2:  ['T0', 'TB', 'TH', 'Pointing次數', 'TB次數', 'TH次數', '反應等級'],
        3:  ['T0', 'TB', 'TH', 'Pointing次數', 'TB次數', 'TH次數', '反應等級'],
        4:  ['T0', 'TB', 'TH', 'Pointing次數', 'TB次數', 'TH次數', '反應等級'],
        5:  ['T0', 'TB', 'TH', 'Pointing次數', 'TB次數', 'TH次數', '反應等級'],
        6:  ['T0', 'TB', 'TH', 'Pointing次數', 'TB次數', 'TH次數', '反應等級'],
        7:  ['T0', 'TB', 'TH', 'Pointing次數', 'TB次數', 'TH次數', '反應等級'],
        8:  ['T0', 'TH', 'TH次數', '反應等級'], 
        9:  ['T0', 'TB', 'TH', 'Pointing次數', 'TB次數', 'TH次數', '反應等級'],
        10: ['T0', 'TB', 'TH', 'Pointing次數', 'TB次數', 'TH次數', '反應等級']
    }
    
    row_data = {'影片編號': vid, '狀態': '完成'}
    
    for stage, cols in stages_schema.items():
        # 初始化預設值
        level, t0, tb, th = 'x', 'x', 'x', 'x'
        pt_cnt, tb_cnt, th_cnt = 0, 0, 0
        
        # 尋找該 Stage 的區塊與反應等級擷取
        block_pattern = rf"Stage\s+{stage}\s+--.*?反應等級\s+([A-Za-z]+).*?(?=Stage\s+\d+\s+--|={40}|$)"
        match = re.search(block_pattern, summary_text, re.DOTALL)
        
        if match:
            level = match.group(1)
            block = match.group(0)
            
            t0_m = re.search(r'T0\s*=\s*(.*?)\n', block)
            if t0_m:
                val = t0_m.group(1).strip()
                if 'not achieved' not in val.lower() and '--' not in val:
                    t0 = val.split()[0]
                    
            tb_m = re.search(r'TB\s*=\s*(.*?)\n', block)
            if tb_m:
                val = tb_m.group(1).strip()
                if 'not achieved' not in val.lower() and '--' not in val:
                    tb = val.split()[0]
                    cnt_m = re.search(r'x(\d+)', val)
                    if cnt_m: tb_cnt = int(cnt_m.group(1))
                    
            th_m = re.search(r'TH\s*=\s*(.*?)\n', block)
            if th_m:
                val = th_m.group(1).strip()
                if 'not achieved' not in val.lower() and '--' not in val:
                    th = val.split()[0]
                    cnt_m = re.search(r'x(\d+)', val)
                    if cnt_m: th_cnt = int(cnt_m.group(1))
                    
            pt_m = re.search(r'Pointing\s*=\s*(.*?)\n', block)
            if pt_m:
                val = pt_m.group(1).strip()
                if 'not detected' not in val.lower() and '--' not in val:
                    cnt_m = re.search(r'x(\d+)', val)
                    if cnt_m: pt_cnt = int(cnt_m.group(1))

        # 次級事件回溯邏輯 (處理時間碰撞)
        stage_log_pattern = rf"Stage change -> {stage}\b(.*?)(?=Stage change ->|$)"
        collision = False
        
        if stage != 8:
            if tb != 'x' and th != 'x' and tb == th:
                collision = True
            elif tb == 'x' and t0 != 'x' and th != 'x' and t0 == th:
                collision = True
        else:
            if t0 != 'x' and th != 'x' and t0 == th:
                collision = True

        if collision:
            stage_log_match = re.search(stage_log_pattern, event_log_text, re.DOTALL)
            if stage_log_match:
                stage_log = stage_log_match.group(1)
                th2_m = re.search(r'\[\s*([\d\.]+)s\s*\]\s*TH#2\b', stage_log)
                if th2_m:
                    th = th2_m.group(1) + 's'

        # ========================================================
        # 將解析結果映射至字典 (Python 3.7+ 將嚴格依照此賦值順序建立 DataFrame 的行向量)
        # ========================================================
        if stage != 8:
            row_data[f'Stage{stage}_T0'] = t0
            row_data[f'Stage{stage}_TB'] = tb
            row_data[f'Stage{stage}_TH'] = th
            row_data[f'Stage{stage}_Pointing次數'] = pt_cnt
            row_data[f'Stage{stage}_TB次數'] = tb_cnt
            row_data[f'Stage{stage}_TH次數'] = th_cnt
            row_data[f'Stage{stage}_反應等級'] = level   # 移至最後一位
        else: # Stage 8
            row_data[f'Stage{stage}_T0'] = t0
            row_data[f'Stage{stage}_TH'] = th
            row_data[f'Stage{stage}_TH次數'] = th_cnt
            row_data[f'Stage{stage}_反應等級'] = level   # 移至最後一位
            
    return row_data

def main():
    parser = argparse.ArgumentParser(description="批次處理 Gaze Tracking 的所有 txt 檔案 (Schema欄位重新排序版)")
    parser.add_argument('-d', '--dir', type=str, default=None, help="指定包含 txt 檔案的資料夾路徑")
    args = parser.parse_args()

    if args.dir:
        target_path_str = args.dir
    else:
        print("請輸入包含 txt 檔案的資料夾路徑 (直接按下 Enter 為當前目錄):")
        target_path_str = input("路徑: ").strip()
        if not target_path_str:
            target_path_str = "./"

    target_dir = Path(target_path_str)

    if not target_dir.is_dir():
        print(f"[Error] 系統無法找到該目錄: {target_dir.resolve()}")
        return

    txt_files = list(target_dir.glob("*.txt"))
    
    if not txt_files:
        print(f"[Warning] 目錄 {target_dir.resolve()} 中未發現任何 .txt 檔案。")
        return
        
    print(f"正在處理 {len(txt_files)} 個檔案...")
    
    all_rows = []
    for f_path in txt_files:
        try:
            row = parse_gaze_log(f_path)
            all_rows.append(row)
            print(f"  - 成功解析: {f_path.name}")
        except Exception as e:
            print(f"  - 解析失敗: {f_path.name}, 錯誤訊息: {e}")
            
    df = pd.DataFrame(all_rows)
    
    # 數值排序過濾器
    def extract_numeric_key(series):
        return series.apply(
            lambda x: int(re.search(r'\d+', str(x)).group()) if re.search(r'\d+', str(x)) else float('inf')
        )
    
    df = df.sort_values(by='影片編號', key=extract_numeric_key).reset_index(drop=True)
    
    output_path = target_dir / "txt_資料統整.csv"
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n[Success] 資料轉換與排序處理完成！結果已存至:\n{output_path.resolve()}")

if __name__ == "__main__":
    main()