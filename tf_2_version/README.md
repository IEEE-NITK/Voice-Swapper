Commands that are to be known to convert audio from Souce A to Target B

1. Run test.py to convert A to B:
`python test.py --test_dir <dir_containing_A_audio> --output_dir <path_to_output_dir> --mceps_dir 
<path_to_mcep_dir> --weight_dir <path_to_weight_dir>`

2. Link for pretrained weights: 
https://drive.google.com/file/d/1ZpFrG88U3QULD8PaqnNLvhScbFCqTB8y/view?usp=sharing

3. Take audio from source A and preprocess it with
`python ./tf_2_version/preprocess.py --train_A_dir ./data/evaluation_all/other_test_voices/ 
--train_B_dir <path_to_B>


