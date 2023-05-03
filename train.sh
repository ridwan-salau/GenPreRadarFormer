

python ./tools/train.py \
--config ./configs/MaxVIT2.py \
--data_dir ./Pickle0 \
--log_dir ./log/ \
--vis_train 0 \
--validate 6
--batch_size $1
