python3 draw_initial_points.py --sequence_path '/srv/fast1/n.anoshina/DefReg/data/SeqB/' --save_path '/srv/fast1/n.anoshina/DefReg/data/dots_drawing/'

python3 apply_deformations2seq.py --sequence_path '/srv/fast1/n.anoshina/DefReg/data/dots_drawing/init_' --prediction_path '/srv/fast1/n.anoshina/DefReg/predictions/test_exp1/defregnet_points_error_l2/'  --save_path '/srv/fast1/n.anoshina/DefReg/data/dots_drawing/proposed/'

python3 calculate_baseline_seq.py --sequence_path '/srv/fast1/n.anoshina/DefReg/data/dots_drawing/init_' --base_prediction_path '/srv/fast1/n.anoshina/DefReg/data/elastic_deformations/'  --save_path '/srv/fast1/n.anoshina/DefReg/data/dots_drawing/elastic/'

python3 EPE_vizualize_deformations.py --sequence_path '/srv/fast1/n.anoshina/DefReg/data/SeqB/SeqB1.tif' --predicted_deformation_path '/srv/fast1/n.anoshina/DefReg/predictions/test_exp1/defregnet_points_error_l2/deformations/' --predicted_theta_path '/srv/fast1/n.anoshina/DefReg/predictions/test_exp1/defregnet_points_error_l2/thetas/' --base_deformation_path '/srv/fast1/n.anoshina/DefReg/data/elastic_deformations/numpy/' --predicted_sequence_path '/srv/fast1/n.anoshina/DefReg/data/dots_drawing/proposed/' --base_sequence_path '/srv/fast1/n.anoshina/DefReg/data/dots_drawing/elastic/' --save_drawing_path '/srv/fast1/n.anoshina/DefReg/EPE_results/SeqB1/viz/'
