import os
import re
import argparse
from pathlib import Path
import pandas as pd

def parse_gaze_log(file_path):
    """
    解析單一 txt 檔案的視線追蹤紀錄，並針對相同時間戳的 TB/TH/T0 進行次級事件回溯。
    傳入的 file_path 為 pathlib.Path 物件。
    """
    vid = file_path.stem
    text = file_path.read_text(encoding='utf-8')
        
    # 如果文本內含有更精確的路徑資訊，則透過 Regex 校正編號
    vid_match = re.search(r'Video:.*[/\\](\d+)\.mp4', text)
    if vid_match:
        vid = vid_match.group(1)

    # 將文本分割為 Summary 與 Full Event Log 兩部分
    parts = text.split('=== Full Event Log ===')
    summary_text = parts[0]
    event_log_text = parts[1] if len(parts) > 1 else ""

    # 定義目標特徵的 Schema
    stages_schema = {
        6:  ['T0', 'TB', 'TH', 'Pointing次數', 'TB次數', 'TH次數'],
        7:  ['T0', 'TB', 'TH', 'Pointing次數', 'TB次數', 'TH次數'],
        8:  ['T0', 'TH', 'TH次數'],
        9:  ['T0', 'TB', 'TH', 'Pointing次數', 'TB次數', 'TH次數'],
        10: ['T0', 'TH', 'TH次數']
    }
    
    row_data = {'影片編號': vid, '狀態': '完成'}
    
    for stage, cols in stages_schema.items():
        t0, tb, th = 'x', 'x', 'x'
        pt_cnt, tb_cnt, th_cnt = 0, 0, 0
        
        # 1. 於 Summary 區塊進行初次特徵擷取
        block_pattern = rf"Stage\s+{stage}\s+--.*?(?=Stage\s+\d+\s+--|={40}|$)"
        match = re.search(block_pattern, summary_text, re.DOTALL)
        
        if match:
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

        # ========================================================
        # 2. 次級特徵回溯邏輯 (Fallback Mechanism for Temporal Collision)
        # ========================================================
        # 定義搜尋 Event Log 中該 Stage 區間的正則表達式
        stage_log_pattern = rf"Stage change -> {stage}\b(.*?)(?=Stage change ->|$)"
        
        if stage in [6, 7, 9]:
            # 若 TB 與 TH 發生碰撞
            if tb != 'x' and th != 'x' and tb == th:
                stage_log_match = re.search(stage_log_pattern, event_log_text, re.DOTALL)
                if stage_log_match:
                    stage_log = stage_log_match.group(1)
                    # 於該區間內搜尋 TH#2 的時間戳，例如: [220.9s] TH#2:
                    th2_m = re.search(r'\[\s*([\d\.]+)s\s*\]\s*TH#2\b', stage_log)
                    if th2_m:
                        th = th2_m.group(1) + 's' # 覆寫 TH 為第二次紀錄
                        
        elif stage in [8, 10]:
            # 若 T0 與 TH 發生碰撞
            if t0 != 'x' and th != 'x' and t0 == th:
                stage_log_match = re.search(stage_log_pattern, event_log_text, re.DOTALL)
                if stage_log_match:
                    stage_log = stage_log_match.group(1)
                    # 於該區間內搜尋 TH#2 的時間戳
                    th2_m = re.search(r'\[\s*([\d\.]+)s\s*\]\s*TH#2\b', stage_log)
                    if th2_m:
                        th = th2_m.group(1) + 's' # 覆寫 TH 為第二次紀錄

        # 3. 將最終解析結果 Mapping 至輸出資料集
        if stage in [6, 7, 9]:
            row_data[f'Stage{stage}_T0'] = t0
            row_data[f'Stage{stage}_TB'] = tb
            row_data[f'Stage{stage}_TH'] = th
            row_data[f'Stage{stage}_Pointing次數'] = pt_cnt
            row_data[f'Stage{stage}_TB次數'] = tb_cnt
            row_data[f'Stage{stage}_TH次數'] = th_cnt
        else: # Stage 8, 10
            row_data[f'Stage{stage}_T0'] = t0
            row_data[f'Stage{stage}_TH'] = th
            row_data[f'Stage{stage}_TH次數'] = th_cnt
            
    return row_data

def main():
    parser = argparse.ArgumentParser(description="批次處理 Gaze Tracking 的所有 txt 檔案，並處理時間碰撞問題")
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
        print(f"[Error] 系統無法找到該目錄，或該路徑無效: {target_dir.resolve()}")
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
    
    # 強制將字串提取為數字並進行數值排序 (Numeric Sorting)
    def extract_numeric_key(series):
        return series.apply(
            lambda x: int(re.search(r'\d+', str(x)).group()) if re.search(r'\d+', str(x)) else float('inf')
        )
    
    df = df.sort_values(by='影片編號', key=extract_numeric_key).reset_index(drop=True)
    
    output_path = target_dir / "txt_資料統整.csv"
    
    # 寫入 CSV 檔案
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n[Success] 資料轉換與排序處理完成！結果已存至:\n{output_path.resolve()}")

if __name__ == "__main__":
    main()
