import ants
import numpy as np
import os
import matplotlib.pyplot as plt
import sys

def resample_to_isotropic_1mm(input_path):
    """
    將影像重取樣成 1 mm isotropic，保持相同物理範圍（會改變輸出影像維度！）
    interp_type=1 為線性插值
    """
    image = ants.image_read(input_path)
    resampled = ants.resample_image(
        image,
        (1, 1, 1),
        use_voxels=False,
        interp_type=1
    )
    return resampled

def make_slice(arr: np.ndarray, output_root: str):
    """
    只儲存特定範圍的切片：
      coronal:   040~190
      sagittal:  035~160
      axial:     041~130
    並將每張切片上下翻轉 + 左右翻轉 + 轉置，儲存成 PNG。
    """
    os.makedirs(output_root, exist_ok=True)
    directions = ['sagittal', 'coronal', 'axial']
    slice_ranges = {
        'sagittal': (40, 160),
        'coronal': (40, 190),
        'axial': (40, 130)
    }

    for axis, name in enumerate(directions):
        dir_path = os.path.join(output_root, name)
        os.makedirs(dir_path, exist_ok=True)

        start, end = slice_ranges[name]
        start = max(start, 0)
        end = min(end, arr.shape[axis])

        for i in range(start, end + 1):
            if axis == 0:
                slice_img = arr[i, :, :]
            elif axis == 1:
                slice_img = arr[:, i, :]
            else:
                slice_img = arr[:, :, i]

            slice_img = slice_img.T
            slice_img = np.flipud(slice_img)
            slice_img = np.fliplr(slice_img)
            i = i - 40
            out_png = os.path.join(dir_path, f"{i:03d}.png")
            plt.imsave(out_png, slice_img, cmap='gray')
        print(f"{name} 切割完畢 ({start}~{end})")

if __name__ == "__main__":
    input_path = sys.argv[1]
    output_dir = sys.argv[2]

    resampled = resample_to_isotropic_1mm(input_path)
    arr = resampled.numpy()

    make_slice(arr, output_dir)
