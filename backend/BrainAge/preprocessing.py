import SimpleITK as sitk
from antspynet.utilities import brain_extraction
import sys
import ants
import numpy as np
import os
def ants_2_itk(image):
    imageITK = sitk.GetImageFromArray(image.numpy().T)
    imageITK.SetOrigin(image.origin)
    imageITK.SetSpacing(image.spacing)
    imageITK.SetDirection(image.direction.reshape(9))
    return imageITK

def itk_2_ants(image):
    image_ants = ants.from_numpy(sitk.GetArrayFromImage(image).T, 
                                 origin=image.GetOrigin(), 
                                 spacing=image.GetSpacing(), 
                                 direction=np.array(image.GetDirection()).reshape(3, 3))
    return image_ants
##################### 去顱骨 #####################
def ants_skull_stripping(input_image):
    # 用 U-net 生成機率圖
    prob_brain_mask = brain_extraction(input_image,modality='t1',verbose=False)
    # 將機率圖轉換為遮罩
    brain_mask = ants.get_mask(prob_brain_mask,low_thresh=0.5)
    # 套用遮罩
    masked_image = ants.mask_image(input_image,brain_mask)
    # 輸出檔案
    #masked_image.to_file(r'lab/IXI_brain_test.nii')
    return masked_image
#################### MNI配準 #####################
def sitk_mni_registration(input_image,template_image):
    # 宣告初始變換
    initial_transform = sitk.CenteredTransformInitializer(
        template_image,
        input_image,
        sitk.Euler3DTransform(),  # 使用剛體變換
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    # 宣告配準方法
    registration_method = sitk.ImageRegistrationMethod()
    
    # 設定配準方法
    # 初始變換
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    # 評估函數：Mattes Mutal Information
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    # 設定優化器：Gradient Descent
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, 
                                                      convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift() #特別重要
    # 設定插值方法：Linear
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    # 執行配準
    final_transform = registration_method.Execute(template_image, input_image)
    # 套用配準
    registered_image = sitk.Resample(input_image, template_image, final_transform, 
                                 sitk.sitkLinear, 0.0, input_image.GetPixelID())
    # 輸出
    return registered_image
#################### 前處理主程式 #####################
def preprocessing(nii_file_path):
    try:
        print("開始前處理:", nii_file_path)
        # 讀取 MNI 模板
        # 取得目前這支腳本的所在資料夾
        base_dir = os.path.dirname(os.path.abspath(__file__))
        abs_mni_path = os.path.join(base_dir, 'mni152-s.nii')
        template_image = sitk.ReadImage(abs_mni_path, sitk.sitkFloat32)
        print("MNI 模板讀檔成功")
        
        # 讀取輸入檔案
        input_image = ants.image_read(nii_file_path)
        print("讀檔成功")
        
        # 去顱骨 (01)
        masked_image = ants_skull_stripping(input_image)
        print("去顱骨成功")
        
        # MNI 配準 (02)
        input_image_itk = ants_2_itk(masked_image)
        registered_image = sitk_mni_registration(input_image_itk, template_image)
        print("配準成功")
        
        # 強度歸一化
        input_image_ants = itk_2_ants(registered_image)
        truncated_img = ants.iMath_truncate_intensity(input_image_ants, 0.05, 0.95)
        normalized_img = ants.iMath_normalize(truncated_img)
        img_uint8 = ((normalized_img - normalized_img.min()) / (normalized_img.max() - normalized_img.min()) * 255).astype('uint8')
        print("強度歸一成功")
        
        # 儲存結果
        filename = os.path.basename(nii_file_path)
        output_path = 'ppResult'
        abs_output_dir = os.path.join(base_dir,output_path) 
        os.makedirs(abs_output_dir, exist_ok=True)
        abs_output_dir = os.path.join(abs_output_dir, filename)
        ants.image_write(img_uint8, abs_output_dir)
        print("存檔成功")
        return abs_output_dir
    except Exception as e:
        print("前處理失敗:", e)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("請提供要處理的 nii.gz 檔案路徑")
        sys.exit(1)
    nii_file_path = sys.argv[1]
    output_path = preprocessing(nii_file_path)
    # 將結果路徑輸出到標準輸出，供呼叫端取得
    print(output_path)