"""Parses the scored tsv and produces values and visualizations.

The parsed tsv is the output of generate_scored_mturk_tsv.py. Prior to running
this script, bad annotators should be removed from this tsv through
annotator_statistics.py and manual editing of the tsv.

Typical usage example:
    python parse_scored_tsv.py --tsv_dir output --trials 1000
    --reject_key tracker_confidence_trunc"""
from __future__ import print_function
import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import matplotlib.patches as patches
import numpy as np
import argparse

# This was written with a different configuration than I typically use.
# And I have no qualms about disabling no-member
# I'm also not sure when to use != vs not ==, but I'm not going to
# change it at this point.
# pylint: disable=bad-indentation, no-member, unneeded-not
PARSER = argparse.ArgumentParser(description="Parse a mturk scored TSV file.")
PARSER.add_argument("--tsv_dir", type=str,
                    help="The directory used as out_dir "+
                    "in generate_scored_mturk_csv.py")
PARSER.add_argument("--trials", type=int, default=2,
                    help="How many trials to run.")
PARSER.add_argument("--reject_key", type=str, default="iou_mean_all_trunc",
                    help="The column used for re-query.")
PARSER.add_argument("--ae_key", type=str, default="additional_error_trunc",
                    help="The column used for additional error.")
PARSER.add_argument("--smart_conf", action="store_true",
                    help="Calculate RMAE using tracker confidence as the "+
                    "selection function.")
ARGS = PARSER.parse_args()

tsv_file = ARGS.tsv_dir + "/summary.tsv"


additional_errors = []
start_ious = []
start_iou_scatter_dict = {}
annots_dict = {}

for i in range(10):
    start_iou_scatter_dict[i] = []

# Put the data from the TSV file in a dict of dicts.
# outer dict is the video name
# inner dict is the annotator as the key, the row as the value
with open(tsv_file, 'r') as tsv_file:
    reader = csv.DictReader(tsv_file, delimiter="\t")
    for row in reader:
        # convert what we can to float here.
        for key in row.keys():
            try:
                row[key] = float(row[key])
            except ValueError:
                pass
        if row['name'] not in annots_dict:
            annots_dict[row['name']] = {}
        annots_dict[row['name']][row['RandomID']] = row

# we run a number of trials specified by the second command line argument.
# these are lists-of-lists with the mean additional error at each coverage.
trial_maes = []
trial_replacement_maes = []
trial_smart_replacement_maes = []
# and this is a list of floats containing the AMAE at each trial.
trial_amaes = []
trial_replacement_amaes = []
trial_smart_replacement_amaes = []

# Used for calculating % below
additional_errors_for_count = []

# run ARGS.trials trials.
for which_trial in range(ARGS.trials):
    trial_additional_errors = []
    # First we build primary and secondary datasets
    first_dataset = {}
    second_dataset = {}

    # loop through every video
    for video in annots_dict.keys():
        # Select two initializations for every video.
        annotator_choices = annots_dict[video].keys()
        chosen_annotators = np.random.choice(
            annotator_choices, 2, replace=False)
        # And save the chosen annotations in first and second query dicts.
        first_dataset[video] = annots_dict[video][chosen_annotators[0]]
        second_dataset[video] = annots_dict[video][chosen_annotators[1]]
        # Keep track of all first query additional errors.
        trial_additional_errors.append(float(first_dataset[video][ARGS.ae_key]))

    # List of lists for calculating statistics over.
    additional_errors_for_count.append(trial_additional_errors)
    # And do sorting for acceptance order.
    if not ARGS.reject_key == "additional_error" and\
       not ARGS.reject_key == "additional_error_trunc":
        sorted_first_dataset = sorted(
            first_dataset.items(), key=lambda x: x[1][ARGS.reject_key],
            reverse=True)
    else:
        sorted_first_dataset = sorted(
            first_dataset.items(), key=lambda x: x[1][ARGS.reject_key],
            reverse=False)

    # Find the additional errors at every coverage
    additional_errors_base = []
    # replacement automatically swaps the first and second anntoations
    additional_errors_replacement = []
    # smart replacement swaps in whichever has the better score.
    additional_errors_smart_replacement = []
    # iou replacement simulates a fully crowdsourced method, no reject_key
    additional_errors_iou_replacement = []

    # Dataset is sorted by the reject key, so iterate through the list
    for element in sorted_first_dataset:
        data = element[1]
        vid_2 = second_dataset[data['name']]

        # get the additional error for the first dataset
        base_additional_error = data[ARGS.ae_key]
        # Additional error has a min of zero.
        base_additional_error = base_additional_error * float(
            base_additional_error > 0)
        # Repeat for the second dataset.
        swap_additional_error = vid_2[ARGS.ae_key]
        swap_additional_error = swap_additional_error * float(
            swap_additional_error > 0)

        # save additional errors for no and  naive replacement.
        additional_errors_base.append(base_additional_error)
        additional_errors_replacement.append(swap_additional_error)

        # save additional
        # smart replacement can reject the new annotation
        # If we're using tracker confidence as the selection function...
        if ARGS.smart_conf:
            # use the higher tracker confidence.
            if data['tracker_confidence'] > vid_2['tracker_confidence']:
                additional_errors_smart_replacement.append(
                    base_additional_error)
            else:
                additional_errors_smart_replacement.append(
                    swap_additional_error)
        else:
            # Otherwise, use the better score.
            if not ARGS.reject_key == "additional_error" and\
               not ARGS.reject_key == "additional_error_trunc":
                if data[ARGS.reject_key] > vid_2[ARGS.reject_key]:
                    additional_errors_smart_replacement.append(
                        base_additional_error)
                else:
                    additional_errors_smart_replacement.append(
                        swap_additional_error)
            else:
                if data[ARGS.reject_key] < vid_2[ARGS.reject_key]:
                    additional_errors_smart_replacement.append(
                        base_additional_error)
                else:
                    additional_errors_smart_replacement.append(
                        swap_additional_error)

    # calculate the mean additional error at each step
    mae_no_replacement = []
    mae_replacement = []
    mae_smart_replacement = []

    for i in range(1, len(additional_errors_base)+1):
        mae_no_replacement.append(np.array(additional_errors_base[:i]).mean())

        # Accept the base (one-annotation) versions with highest ratings, then
        # use the replacement annotaitons.
        mae_replacement.append(np.array(
            additional_errors_base[:i] + additional_errors_replacement[i:])\
            .mean())
        mae_smart_replacement.append(np.array(
            additional_errors_base[:i] +\
            additional_errors_smart_replacement[i:]).mean())

    # Calc AMAE from the MAE
    amae_no_replacement = np.array(mae_no_replacement).mean()
    amae_replacement = np.array(mae_replacement).mean()
    amae_smart_replacement = np.array(mae_smart_replacement).mean()

    # save for this trial.
    trial_maes.append(mae_no_replacement)
    trial_amaes.append(amae_no_replacement)
    trial_replacement_maes.append(mae_replacement)
    trial_replacement_amaes.append(amae_replacement)
    trial_smart_replacement_maes.append(mae_smart_replacement)
    trial_smart_replacement_amaes.append(amae_smart_replacement)

# calc mean and standard error
print("Normal AMAE: ")
print(
    np.array(trial_amaes).mean(),
    np.array(trial_amaes).std()/np.sqrt(len(trial_amaes)))
print("Replacement AMAE: ")
print(np.array(trial_replacement_amaes).mean(),
      np.array(trial_replacement_amaes).std()/np.sqrt(
          len(trial_replacement_amaes)))
print("Smart Replacement AMAE: ")
print(np.array(trial_smart_replacement_amaes).mean(),
      np.array(trial_smart_replacement_amaes).std()/np.sqrt(
          len(trial_smart_replacement_amaes)))

for acceptable_error in range(10):
    lt_acceptable_error = []
    for trial in additional_errors_for_count:
        lt_acceptable_error.append(
            (np.array(trial) <= acceptable_error/10.).mean())

    print("Proportion lt " + str(acceptable_error/10.) + " additional error: ")
    print(np.array(lt_acceptable_error).mean(),
          np.array(lt_acceptable_error).std())
with open(ARGS.tsv_dir + "/" + ARGS.reject_key + "_"+\
          str(ARGS.smart_conf)+".csv", "w") as csv_file:
    csv_file.write("Coverage,MAE Mean Normal,MAE StdErr Normal,"+\
                   "MAE Mean Replace,MAE StdErr Replace,MAE Mean SReplace,"+
                   "MAE StdErr Sreplace\n")
    # loop through every coverage
    for i in range(len(mae_no_replacement)):
        cur_list_base = []
        cur_list_replace = []
        cur_list_smart_replace = []
        # then through every sample
        for j in range(len(trial_maes)):
            cur_list_base.append(trial_maes[j][i])
            cur_list_replace.append(trial_replacement_maes[j][i])
            cur_list_smart_replace.append(trial_smart_replacement_maes[j][i])
        csv_file.write(str(float(i+1)/len(mae_no_replacement))+","+
                       str(np.array(cur_list_base).mean())+","+
                       str(np.array(cur_list_base).std()/np.sqrt(
                           len(cur_list_base)))+","+
                       str(np.array(cur_list_replace).mean())+","+
                       str(np.array(cur_list_replace).std()/np.sqrt(
                           len(cur_list_replace)))+","+
                       str(np.array(cur_list_smart_replace).mean())+","+
                       str(np.array(cur_list_smart_replace).std()/np.sqrt(
                           len(cur_list_smart_replace)))+"\n")

# Loop through every video and every annotator.
for video in annots_dict.keys():
    for annotator in annots_dict[video].keys():
        # For clarity/brevity, pull from dict here.
        row = annots_dict[video][annotator]

        # for generating histograms.
        start_ious.append(float(row['iou_true']))
        additional_errors.append(float(row[ARGS.ae_key]))
        # start_iou_scatter_dict is a list of length 10, corresponding
        # to each of the bins. Find the correct index and add the current
        # video's additional error to it.
        boxplot_key = np.floor(start_ious[-1]*10)
        start_iou_scatter_dict[boxplot_key].append(additional_errors[-1])

# Create histogram of start ious (fig 3)
plt.cla()
hist = plt.hist(start_ious, np.arange(0, 1.01, 0.1),
                weights=np.ones(len(start_ious))/len(start_ious))
plt.xlabel("Annotation IoU")
plt.ylabel("Percent of Samples")
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.savefig(ARGS.tsv_dir+"/hist-"+ARGS.ae_key+".pdf")
plt.cla()

# Create the boxplot (Figure 7)
ax = plt.gca()
boxplot_list = []

# The boxplot wants a list of arrays, so generate it.
for key, value in start_iou_scatter_dict.items():
    boxplot_list.append(np.array(value))
# Then build the boxplot.
ax.boxplot(boxplot_list, positions=np.arange(0, .99, 0.1)+.05,
           widths=0.075, manage_xticks=False)
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.xticks(np.arange(0, 1.01, 0.1))
plt.xlabel("Start IoU (binned)")
plt.ylabel("Additional Error")
# Draw the rectangle.
rect = patches.Rectangle((0.5, 0.36),
                         0.48, 0.39, fill=False, edgecolor='r', ls='--')
plt.text(0.5, 0.75, "Successful Initialization, Poor Performance",
         horizontalalignment='left', verticalalignment='bottom',
         color='r', fontsize='8')
ax = plt.gca()
ax.add_patch(rect)
plt.savefig(ARGS.tsv_dir+"/box-"+ARGS.ae_key+".pdf")
plt.cla()
