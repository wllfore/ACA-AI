from monai.data import load_decathlon_datalist, decollate_batch, CacheDataset, DataLoader
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ResizeWithPadOrCropd,
    Invertd,
    SaveImaged,
    AsDiscreted
)
from monai.inferers import sliding_window_inference
import SimpleITK as sitk
from models.create_model import create_model_custom
from tqdm import tqdm
import torch
import numpy as np
import os

import SimpleITK as sitk
from medpy import metric

### set config paras
data_dir = '/path_to_image_data'
model_path = './weights/aca_seg_model.pth'
pred_mask_save_dir = './output'

img_patch_size = (96, 96, 64)
pixelspacing = (1.0, 1.0, 1.0)

def pred():
   
    model = create_model_custom(
        'uniformer_small_IL',
        pretrained=False,
        num_classes=4,
        num_phase=1,
        drop_rate=0.0,
        drop_path_rate=None,
        drop_block_rate=None,
        bn_momentum=None,
        bn_eps=None,
        scriptable=False)
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.cuda()

    val_transforms = Compose(
        [
            LoadImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),
            Orientationd(keys="image", axcodes="RAS"),
            Spacingd(
                keys="image",
                pixdim=pixelspacing,
                mode="bilinear",
            ),
            ScaleIntensityRanged(keys="image", a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys="image", source_key="image"),
        ]
    )

    post_transforms = Compose(
        [
            Invertd(
                keys="pred",
                transform=val_transforms,
                orig_keys="image",
                nearest_interp=False,
                to_tensor=True
            ),
            AsDiscreted(
                keys="pred",
                argmax=True,
                # dtype=np.uint8
            ),
            SaveImaged(
                keys="pred",
                output_dir=pred_mask_save_dir,
                resample=False,
                output_dtype=np.uint8
            ),
        ]
    )

    datalist = []
    file_list = os.listdir(data_dir)
    for fs in file_list:
        image_file = os.path.join(data_dir, fs)
        datalist.append({'image': image_file})

    test_ds = CacheDataset(data=datalist, transform=val_transforms, cache_rate=0.0, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    test_iterator = tqdm(test_loader)

    with torch.no_grad():
        for batch_input in test_iterator:
            x = batch_input["image"].cuda()
            outputs = sliding_window_inference(x, img_patch_size, 4, model)
            batch_input["pred"] = outputs.detach().cpu()
            batch_input = [post_transforms(i) for i in decollate_batch(batch_input)]

if __name__ == "__main__":
    
    ### do AA and AG segmentation
    pred()
