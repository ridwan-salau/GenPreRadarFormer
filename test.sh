set -e

run_folder=maxvit2_3579-20230505-032236
epoch=30

python ./tools/test.py \
--config ./configs/MaxVIT2.py \
--data_dir ./Pickle0/ \
--checkpoint ./log/$run_folder/epoch_${epoch}_final.pkl \
--res_dir ./results/ ;


result_dir=${run_folder}_${epoch}
python ./tools/format_transform/convert_rodnet_to_rod2021.py \
--result_dir ./results/${run_folder} \
--final_dir ./final/${result_dir}/ ;


# cd ./final_${result_dir}/

python -c "import shutil;shutil.make_archive('./result', 'zip', './final/${result_dir}/')"

python tools/eval.py  --res_dir results/${run_folder}/ --data_root ../data/ --gt_dir ../data/annotations/train/