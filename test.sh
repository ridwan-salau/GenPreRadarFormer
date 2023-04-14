set -e

run_folder=maxvit2_3579-20230409-202229
epoch=30

python ./tools/test.py \
--config ./configs/MaxVIT2.py \
--data_dir ./Pickle0/ \
--checkpoint ./log/$run_folder/epoch_$epoch\_final.pkl \
--res_dir ./results/ ;




python ./tools/format_transform/convert_rodnet_to_rod2021.py \
--result_dir ./results/$run_folder/ \
--final_dir ./results/$run_folder/ ;


cd ./results/$run_folder/

zip ./result_$(date +%Y%m%d_%H%M%S%Z).zip 2019*.txt
