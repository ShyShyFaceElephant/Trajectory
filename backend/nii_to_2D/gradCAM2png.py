import ants
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import os
import sys

def generate_overlay_slices(mri_path: str, cam_path: str, output_dir: str):
    """
    將 MRI 與 Grad-CAM 疊圖，輸出三視角指定範圍內的 PNG 切片。
    檔名格式為 axi_0.png、cor_0.png、sag_0.png。
    """
    print("🔄 開始重取樣與對齊...")
    nii_img = ants.image_read(mri_path)
    cam_img = ants.image_read(cam_path)

    nii_resampled = ants.resample_image(nii_img, (1, 1, 1), use_voxels=False, interp_type=1)
    cam_resampled = ants.resample_image_to_target(cam_img, nii_resampled, interp_type=1)

    nii = nii_resampled.numpy()
    cam = cam_resampled.numpy()

    nii = (nii - np.min(nii)) / (np.max(nii) - np.min(nii))
    cam = np.nan_to_num(cam)
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)

    alpha = 0.6
    threshold = 0.5

    # 自訂切片範圍
    slice_ranges = {
        'gradCAM_sagittal': (40, 160),
        'gradCAM_coronal': (40, 190),
        'gradCAM_axial': (40, 130)
    }

    directions = {
        "gradCAM_axial":    lambda img, i: img[:, :, i],
        "gradCAM_coronal":  lambda img, i: img[:, i, :],
        "gradCAM_sagittal": lambda img, i: img[i, :, :]
    }

    print("🖼️ 開始輸出疊圖 PNG...")
    os.makedirs(output_dir, exist_ok=True)

    for name, slicer in directions.items():
        dir_path = os.path.join(output_dir, name)
        os.makedirs(dir_path, exist_ok=True)

        start, end = slice_ranges[name]

        for j, i in enumerate(range(start, end+1)):
            bg = slicer(nii, i)
            overlay = slicer(cam, i)

            bg_disp = np.fliplr(np.flipud(bg.T))
            overlay_disp = np.fliplr(np.flipud(overlay.T))

            overlay_masked = ma.masked_less(overlay_disp, threshold)

            plt.figure(figsize=(5, 5))
            plt.imshow(bg_disp, cmap='gray', origin='upper')

            cmap = plt.get_cmap('jet')
            cmap.set_bad(alpha=0.0)
            plt.imshow(overlay_masked, cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1), alpha=alpha)

            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(dir_path, f"{i-40:03d}.png"), bbox_inches='tight', pad_inches=0)
            plt.close()

    print(f"✅ 疊圖完成：{output_dir}，每方向切片範圍已套用")

    


if __name__ == "__main__":
    input_path = sys.argv[1]
    cam_path = sys.argv[2]
    output_dir = sys.argv[3]
    generate_overlay_slices(input_path, cam_path, output_dir)
