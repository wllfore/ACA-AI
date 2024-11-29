## Implementation code of the ACA-Seg and ACA-Diag models in the paper: A Clinical AI Study for Automated Diagnosing Adrenocortical Adenoma at Contrast-Enhanced CT
****
## Requirements
* The following setup has been tested on Python 3.9, Ubuntu 20.04.
* Major dependences: pytorch 1.13.1 monai 1.3.1 SimpleITK 2.3.1
****
## Usage
* Download the weight files from the following link, place 'aca_seg_model.pth' in 'ACA-Seg/weights' and 'uniformer_small_k400_8x8_partial.pth' in 'ACA-Diag/weights'.
  Link: https://pan.baidu.com/s/1OjYFctHHQycacdI3raX9SQ   Password: vr3g
* Run 'do_test.sh' in 'ACA-Seg/' to conduct ACA segmentation, need config your data path.
* Run 'do_train.sh' in 'ACA-Diag/' to train the ACA-Diag model and run 'do_test.sh' to test the trained model. 
