import os 
import subprocess

def runSlice(input_path, output_dir):
    """使用指定 Conda 環境執行 nii2png.py，傳入輸入檔與輸出資料夾"""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))  # 這是 nii_to_png.py 的位置
        script_path = os.path.join(base_dir, "nii2png.py")

        result = subprocess.run(
            ["conda", "run", "-n", "brainAge_pp_env", "python", script_path, input_path, output_dir],
            capture_output=True,
            text=True,
            check=True
        )

        msg = result.stdout.strip()
        print("===== 切片執行狀況 =====")
        print(msg)
        print("===== 切片執行結束 =====")
        
    except subprocess.CalledProcessError as e:
        print("切片執行失敗：", e.stderr)
        return None

    print("切片結果已儲存於:", output_dir)
    return output_dir

def runGradCAMSlice(input_path, cam_path, output_dir):
    """使用指定 Conda 環境執行 nii2png.py，傳入輸入檔與輸出資料夾"""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))  # 這是 nii_to_png.py 的位置
        script_path = os.path.join(base_dir, "gradCAM2png.py")

        result = subprocess.run(
            ["conda", "run", "-n", "brainAge_pp_env", "python", script_path, input_path, cam_path, output_dir],
            capture_output=True,
            text=True,
            check=True
        )

        msg = result.stdout.strip()
        print("===== gradCAM切片執行狀況 =====")
        print(msg)
        print("===== gradCAM切片執行結束 =====")
        
    except subprocess.CalledProcessError as e:
        print("切片執行失敗：", e.stderr)
        return None

    print("切片結果已儲存於:", output_dir)
    return output_dir
