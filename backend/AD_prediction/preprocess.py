import ants
import os
from antspynet.utilities import brain_extraction
import SimpleITK as sitk
import numpy as np

#n4 bias field correction=================================================================================

def n4_bias_field_correction(
    image,
    mask=None,
    rescale_intensities=False,
    shrink_factor=4,
    convergence={"iters": [50, 50, 50, 50], "tol": 1e-7},
    spline_param=None,
    return_bias_field=False,
    verbose=False,
    weight_mask=None,
):
    return ants.n4_bias_field_correction(
        image,
        mask=mask,
        shrink_factor=shrink_factor,
        convergence=convergence,
        spline_param=spline_param,
        rescale_intensities=rescale_intensities,
        return_bias_field=return_bias_field,
        verbose=verbose,
        weight_mask=weight_mask
    )


def abp_n4(image, intensity_truncation=(0.025, 0.975, 256), mask=None, usen3=False):
    if (not isinstance(intensity_truncation, (list, tuple))) or (
        len(intensity_truncation) != 3
    ):
        raise ValueError("intensity_truncation must be list/tuple with 3 values")
    
    # Truncate intensities
    outimage = ants.iMath(
        image,
        "TruncateIntensity",
        intensity_truncation[0],
        intensity_truncation[1],
        intensity_truncation[2],
    )
    
    if usen3:
        # Apply N3 bias field correction
        outimage = ants.n3_bias_field_correction(outimage, 4)
        outimage = ants.n3_bias_field_correction(outimage, 2)
    else:
        # Apply N4 bias field correction
        outimage = n4_bias_field_correction(outimage, mask)
    
    return outimage

def n4(original_image_path):#放的是原始影像的路徑
    image = ants.image_read(original_image_path)
    corrected_image = abp_n4(image)
    return corrected_image#輸出n4 偏差場校正的影像

#Nonlinear Registration=================================================================================

def reg(corrected_image,fixed_path):
    #corrected_image也就是n4偏差場校正過後的影像
    #fixed_path是模板路徑這裡是mni_icbm_152(可以自己換)
    fixed_image = ants.image_read(fixed_path)
    
    transform = ants.registration(fixed=fixed_image, moving=corrected_image, 
                              type_of_transform='SyN', verbose=True)
    #type_of_transform可以自行選擇要甚麼變換
    warped_image = ants.apply_transforms(fixed=fixed_image, moving=corrected_image,
                                     transformlist=transform['fwdtransforms'])
    return warped_image


#skull-stripping=================================================================================

def skull(warped_image):
    #warped_image為配準後的影像
    reorient_warped_image = ants.reorient_image2(warped_image, orientation='IAL')
    prob_brain_mask = brain_extraction(reorient_warped_image , modality='t1', verbose=True)
    brain_mask = ants.get_mask(prob_brain_mask, low_thresh=0.5)
    masked = ants.mask_image(reorient_warped_image , brain_mask)
    #輸出masked為去除顱骨後的影像
    return masked

#Min-Max[0,1]=================================================================================

#先將antspy轉成simpleitk
def ants_2_itk(image):
    imageITK = sitk.GetImageFromArray(image.numpy().T)
    imageITK.SetOrigin(image.origin)
    imageITK.SetSpacing(image.spacing)
    imageITK.SetDirection(image.direction.reshape(9))
    return imageITK

def min_max_normalization(masked):
    masked = ants_2_itk(masked)
    normalized_image = sitk.RescaleIntensity(masked, outputMinimum=0.0, outputMaximum=1.0)
    #可自行調整outputMinimum=0.0, outputMaximum=1.0使影像介於[0,1]之間
    return normalized_image

#Resample=================================================================================

def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):
    resampler = sitk.ResampleImageFilter()
    originSize = np.array(itkimage.GetSize(), float)
    originSpacing = np.array(itkimage.GetSpacing(), float)
    newSize = np.array(newSize, float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int32)
    resampler.SetReferenceImage(itkimage)
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)
    return itkimgResampled

def resample(normalized_image):
    #輸入為min-max normalization[0,1]
    resize_image = resize_image_itk(normalized_image,(128,128,128),resamplemethod=sitk.sitkNearestNeighbor)
    #輸出resize_image為尺寸128，(128,128,128)可調
    return resize_image

#主程式=================================================================================
#此程式是一步做到做後，可自行更改順序，如果不會改的話，n4要先其他就可以任意更改位置
#此程式只要3個路徑即可，1.原始影像資料夾、2.腦的模板、3.儲存影像的資料夾
if __name__ == "__main__":
    original_image_folder = r"C:\dementia project\50 years\AD 55 years\003_S_6264.nii"
    #original_image_folder是未處理的原始nii影像資料夾
    fixed_path = r"C:\dementia project\mni_icbm152_t1_tal_nlin_sym_09a.nii"
    #fixed_path就是腦的模板
    save_folder = r"C:\dementia project\50 years\AD 55 years"
    #經過處理過後的儲存資料夾
    #此前處理的順序為n4-->配準-->去除顱骨-->Min-Max[0,1]-->重採樣
    #以下程式為批次處理
    for filename in os.listdir(original_image_folder):
        original_image_path = os.path.join(original_image_folder,filename)
        save_path = os.path.join(save_folder,filename)
        
        corrected_image = n4(original_image_path)
        #145行是n4，裡面放的original_image_path是影像路徑
        warped_image = reg(corrected_image, fixed_path)
        #147行是配準，注意corrected_image放的是張量而非路徑，fixed_path是腦的模板
        masked = skull(warped_image)
        #149行是去顱骨，注意warped_image放的是張量而非路徑
        normalized_image = min_max_normalization(masked)
        #151行是Min-Max normalization，注意masked放的是張量而非路徑
        resize_image = resample(normalized_image)
        #153行是重採樣，注意normalized_image放的是張量而非路徑
        sitk.WriteImage(resize_image, save_path)
        print(f'已經儲存在:{save_path}')
        #最後影像會批次儲存在136行的資料夾裡面
    
