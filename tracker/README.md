# Tracking Experiments
This folder contains the scripts and instructions for replicating the results in the paper after the crowdsourced annotations have been collected. The tracker code is forked from the implementation of Distraction-Aware Siamese RPN available [here](https://github.com/foolwood/DaSiamRPN).

## Setting up Environment
Included in this folder is an exported conda environment (environment.yml). Use the instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) to set up the environment. Activate it with the command:

    conda activate crowdsourcing-effective-initializations

## Training IoU Regressor
The weights for the IoU regressor can be downloaded [here](https://drive.google.com/drive/folders/1XTRm1FXHo1Q7olfZkMNwE8u475SucRVm?usp=sharing), saving you the trouble of training it.

However, if you choose to train it yourself, you can follow these steps:

 1. Install and configure [Weights & Biases](http://wandb.com). The W&B library does not come in the conda environment, and must be installed with pip:

	    pip install --upgrade wandb

 2. Download the [MSCOCO detection dataset](https://cocodataset.org/#detection-2015).
 3. Change the paths on lines 267-270 of `train_iou_prediction.py`.
 4. Execute the training script:
	 
	    python train_iou_prediction.py --batch_size 16 --eval_step 1 --lr 1e-3 --output_dir iou_regressor --patience 10

The weights will be saved as: iou_regressor/iou_regressor/[wandb run id]/checkpoint_best_loss.weights



## Running Experiments/Generating Graphics
The weights for DaSiamRPN are available [here](https://drive.google.com/open?id=1btIkp5pB6aqePQGlMb2_Z7bfPy6XEj6H). I have not included the training scripts in this repository, but they are available from the original author of the [DaSiamRPN implementation](https://github.com/foolwood/DaSiamRPN). In the tracker folder:

1.  Download the OTB100 dataset:
	
	    cd data && sh get_otb_data.sh && cd ..
2.  Generate the various scores (this takes a while, it has to process all the videos three times):

`python generate_scored_mturk_tsv.py --out_dir output --guess_iou_weights [path to guess weights] --mturk_tsv ../mturk_tools/mturk_output.tsv --description_tsv ../mturk_tools/bbox-seed-descriptions.tsv`
3. Find out which annotators you have to filter:

	    python annotator_statistics.py output/summary.tsv
4. Remove every annotator who has >15\% bad annotations from the summary.tsv file (I recommend backing this file up first).
5. Generate the output values by running, as well as Figures 3 and 7, by running the following for all scoring functions of interest, with and without the `--smart_conf` flag:

	    python parse_scored_tsv.py --tsv_dir output --trials 1000 --reject_key [header from csv file] [--smart_conf]
6. Produce the plots for Figures 9-12:
 `python csvs_to_plot.py method_1.csv method_1_name method_2.csv method_2_name [...]`
