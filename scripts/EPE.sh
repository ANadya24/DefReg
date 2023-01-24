EXP="test_exp1/defregnet_points_error_l2"
python3 inference.py inference_config.yaml

SEQ="SeqB1"
# SEQB1
python3 EPE_calculate_initial_errors.py "/srv/fast1/n.anoshina/DefReg/data/SeqB/$SEQ.tif" "/srv/fast1/n.anoshina/DefReg/EPE_results/"

python3 EPE_calculate_baseline_errors.py "/srv/fast1/n.anoshina/DefReg/data/SeqB/$SEQ.tif" "/srv/fast1/n.anoshina/DefReg/EPE_results/"

python3 EPE_calculate_prediction_errors.py --sequence_path "/srv/fast1/n.anoshina/DefReg/data/SeqB/$SEQ.tif" --prediction_path "/srv/fast1/n.anoshina/DefReg/predictions/$EXP/" --use_thetas 1 --save_pickle_path "/srv/fast1/n.anoshina/DefReg/EPE_results/$EXP/$SEQ/proposed_"

python3 EPE_create_graphics.py --baseline_pickle_path "/srv/fast1/n.anoshina/DefReg/EPE_results/elastic_method_bcw_" --initial_error_pickle_path "/srv/fast1/n.anoshina/DefReg/EPE_results/unregistered_" --proposed_pickle_path "/srv/fast1/n.anoshina/DefReg/EPE_results/$EXP/$SEQ/proposed_" --sequence_path "/srv/fast1/n.anoshina/DefReg/data/SeqB/$SEQ.tif" --save_graphics_path "/srv/fast1/n.anoshina/DefReg/EPE_results/$EXP/$SEQ/"


SEQ="SeqB4"
# SEQB4
python3 EPE_calculate_initial_errors.py "/srv/fast1/n.anoshina/DefReg/data/SeqB/$SEQ.tif" "/srv/fast1/n.anoshina/DefReg/EPE_results/"

python3 EPE_calculate_baseline_errors.py "/srv/fast1/n.anoshina/DefReg/data/SeqB/$SEQ.tif" "/srv/fast1/n.anoshina/DefReg/EPE_results/"

python3 EPE_calculate_prediction_errors.py --sequence_path "/srv/fast1/n.anoshina/DefReg/data/SeqB/$SEQ.tif" --prediction_path "/srv/fast1/n.anoshina/DefReg/predictions/$EXP/" --use_thetas 1 --save_pickle_path "/srv/fast1/n.anoshina/DefReg/EPE_results/$EXP/$SEQ/proposed_"

python3 EPE_create_graphics.py --baseline_pickle_path "/srv/fast1/n.anoshina/DefReg/EPE_results/elastic_method_bcw_" --initial_error_pickle_path "/srv/fast1/n.anoshina/DefReg/EPE_results/unregistered_" --proposed_pickle_path "/srv/fast1/n.anoshina/DefReg/EPE_results/$EXP/$SEQ/proposed_" --sequence_path "/srv/fast1/n.anoshina/DefReg/data/SeqB/$SEQ.tif" --save_graphics_path "/srv/fast1/n.anoshina/DefReg/EPE_results/$EXP/$SEQ/"
