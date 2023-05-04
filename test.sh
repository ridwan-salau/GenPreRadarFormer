set -e

run_folder=maxvit2_3579-20230503-140008
epoch=30

python ./tools/test.py \
--config ./configs/MaxVIT2.py \
--data_dir ./Pickle0/ \
--checkpoint ./log/$run_folder/epoch_$epoch\_final.pkl \
--res_dir ./results/ ;



python ./tools/format_transform/convert_rodnet_to_rod2021.py \
--result_dir ./results/${run_folder} \
--final_dir ./final_${run_folder}_${epoch}/ ;


# cd ./final_${run_folder}_${epoch}/

python -c "import shutil;shutil.make_archive('./result', 'zip', './final_${run_folder}_${epoch}/')"
