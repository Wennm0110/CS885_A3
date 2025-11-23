import subprocess
import sys
import os # 引入 os 模組

def run_script(script_name):
    """
    執行指定的 Python 腳本，並讓其輸出直接顯示在當前終端機。
    
    Args:
        script_name (str): 要執行的腳本檔案名稱 (例如 "DQN.py")。
    """
    print(f"\n--- 開始執行 {script_name} ---")
    
    # 檢查檔案是否存在
    if not os.path.exists(script_name):
        print(f"錯誤：找不到腳本檔案 '{script_name}'。請確認檔案在同一個資料夾中。")
        return

    try:
        # 使用 sys.executable 確保我們用的是當前環境的 Python 解釋器
        
        # === 【重要修改】 ===
        # 移除 capture_output=True, text=True, 和 encoding='utf-8'
        # 這會讓子腳本 (DQN.py/DRQN.py) 的 stdout 和 stderr
        # 直接繼承並輸出到這個 run_all.py 腳本的終端機上。
        
        result = subprocess.run(
            [sys.executable, script_name],
            check=True # 保持 check=True，如果腳本執行失敗，這裡仍會拋出錯誤
        )
        
        print(f"\n--- {script_name} 執行完畢 ---")

    except FileNotFoundError:
        print(f"錯誤：找不到 Python 解釋器 '{sys.executable}'。")
    except subprocess.CalledProcessError as e:
        # 如果腳本執行失敗 (例如 DQN.py 內部有語法錯誤)
        print(f"--- {script_name} 執行失敗 ---")
        print("返回狀態碼:", e.returncode)
        # 由於輸出已經直接顯示，我們不需要再印 e.stdout 或 e.stderr
    except KeyboardInterrupt:
        print(f"\n--- {script_name} 被使用者手動中斷 ---")
        # 拋出 KeyboardInterrupt 以便停止整個 run_all.py
        raise
    except Exception as e:
        print(f"執行 {script_name} 時發生未預期的錯誤: {e}")

if __name__ == "__main__":
    # 定義要依序執行的腳本列表
    scripts_to_run = ["DQN.py", "DRQN.py"]
    
    try:
        for script in scripts_to_run:
            run_script(script)
        
        print("\n=== 所有腳本執行完畢 ===")
        print("圖片應已儲存為 dqn_results.png 和 drqn_results.png")
    
    except KeyboardInterrupt:
        print("\n=== 主程式被中斷，已停止執行 ===")