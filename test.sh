set -e

run_test(){
run_folder=maxvit2_3579-20230508-183845
epoch=$1

python ./tools/test.py \
--config ./configs/MaxVIT2.py \
--data_dir ./Pickle0/ \
--data_root ./Pickle0/ \
--checkpoint ./log/$run_folder/epoch_${epoch}_final.pkl \
--res_dir ./results/ ;


result_dir=${run_folder}_${epoch}
python ./tools/format_transform/convert_rodnet_to_rod2021.py \
--result_dir ./results/${result_dir} \
--final_dir ./final/${result_dir}/ ;


# cd ./final_${result_dir}/

python -c "import shutil;shutil.make_archive('./final/${result_dir}', 'zip', './final/${result_dir}/')"

python tools/eval.py  --res_dir results/${result_dir}/ --data_root ../data/ --gt_dir ../data/annotations/train/ > ./final/${result_dir}/eval_result.txt

echo Find the results here: ./final/${result_dir}/eval_result.txt
}

for epoch in {43..43..3}; do
run_test $epoch &
done