## SEQB1
# python3 EPE_calculate_initial_baseline_errors.py '/srv/fast1/n.anoshina/DefReg/data/SeqB/SeqB1.tif' '/srv/fast1/n.anoshina/DefReg/EPE_results/SeqB1/'

# python3 EPE_calculate_prediction_errors.py --sequence_path '/srv/fast1/n.anoshina/DefReg/data/SeqB/SeqB1.tif' --prediction_path '/srv/fast1/n.anoshina/DefReg/predictions/test_exp1/defregnet_points_error_l2/' --use_thetas 1 --save_pickle_path '/srv/fast1/n.anoshina/DefReg/EPE_results/SeqB1/proposed_'

# python3 EPE_create_graphics.py --baseline_pickle_path '/srv/fast1/n.anoshina/DefReg/EPE_results/SeqB1/elastic_method_fwd_' --initial_error_pickle_path '/srv/fast1/n.anoshina/DefReg/EPE_results/SeqB1/unregistered_' --proposed_pickle_path '/srv/fast1/n.anoshina/DefReg/EPE_results/SeqB1/proposed_' --sequence_path '/srv/fast1/n.anoshina/DefReg/data/SeqB/SeqB1.tif' --save_graphics_path '/srv/fast1/n.anoshina/DefReg/EPE_results/SeqB1/'


## SEQB4
python3 EPE_calculate_initial_baseline_errors.py '/srv/fast1/n.anoshina/DefReg/data/SeqB/SeqB4.tif' '/srv/fast1/n.anoshina/DefReg/EPE_results/SeqB4/'

python3 EPE_calculate_prediction_errors.py --sequence_path '/srv/fast1/n.anoshina/DefReg/data/SeqB/SeqB4.tif' --prediction_path '/srv/fast1/n.anoshina/DefReg/predictions/test_exp1/defregnet_points_error_l2/' --use_thetas 1 --save_pickle_path '/srv/fast1/n.anoshina/DefReg/EPE_results/SeqB4/proposed_'

python3 EPE_create_graphics.py --baseline_pickle_path '/srv/fast1/n.anoshina/DefReg/EPE_results/SeqB4/elastic_method_fwd_' --initial_error_pickle_path '/srv/fast1/n.anoshina/DefReg/EPE_results/SeqB4/unregistered_' --proposed_pickle_path '/srv/fast1/n.anoshina/DefReg/EPE_results/SeqB4/proposed_' --sequence_path '/srv/fast1/n.anoshina/DefReg/data/SeqB/SeqB4.tif' --save_graphics_path '/srv/fast1/n.anoshina/DefReg/EPE_results/SeqB4/'