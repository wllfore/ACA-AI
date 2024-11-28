python validate.py \
--data_dir /path_to_image_data \
--val_anno_file /path/test.txt \
--radiomics_fea_file /path/total_radiomics_norm.csv --use_radio 1 \
--model uniformer_small_IL \
--num-classes 3 --b 4 \
--img_size 16 128 128 \
--crop_size 14 112 112 \
--checkpoint /path_to_model \
--results-dir ./output
