"""Plots a number of mae/rmae coverage curves.

Uses the output of parse_scored.tsv as its arguments, and produces Figures
8-11. Note that Fig 11 (tracker confidence as selection function) will need
to be generated separately from Fig 10 (selection function = re-query fn),
since they are output from different runs of parse_scored_tsv.py.

Typical usage:
    python csvs_to_plot.py out_dir/file1.csv "Tracker Confidence"
    out_dir/file2.csv "Cycle Consistency" [...]
"""

import csv
import matplotlib.pyplot as plt
import sys
import numpy as np
from collections import OrderedDict

# This was written with a different configuration than I typically use.
# And I have no qualms about disabling no-member
# change it at this point.
# pylint: disable=bad-indentation, no-member

# Three cases: no replacement, naive replacement, and smart replacement.
# all three have both a a mean and stderr at each coverage.
# The dict is keyed to each filename (i.e., each scoring fn).
normal_dict = OrderedDict()
normal_stderr_dict = {}
replace_dict = {}
replace_stderr_dict = {}
smart_replace_dict = {}
smart_replace_stderr_dict = {}
coverage_dict = {}

# loop through all of the scoring functions (each of which has two arguments
# corresponding to the output csv and the name).
label_dict = {} # The name of each scoring fn.
for i in range(1, len(sys.argv), 2):
    # Read all coverages, stderrs, etc into a list stored in the dictionary
    # location corresponding to the filename.

    # Start by creating those lists.
    cur_file = sys.argv[i]
    normal_dict[cur_file] = []
    normal_stderr_dict[cur_file] = []
    replace_dict[cur_file] = []
    replace_stderr_dict[cur_file] = []
    smart_replace_dict[cur_file] = []
    smart_replace_stderr_dict[cur_file] = []
    coverage_dict[cur_file] = []

    # Open the file and fill in the lists. Each coverage is a list item,
    # and all of this information is in the csv.
    with open(cur_file, 'r') as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            label_dict[cur_file] = sys.argv[i+1]
            normal_dict[cur_file].append(float(row['MAE Mean Normal']))
            normal_stderr_dict[cur_file].append(float(row['MAE StdErr Normal']))
            replace_dict[cur_file].append(float(row['MAE Mean Replace']))
            replace_stderr_dict[cur_file].append(
                float(row['MAE StdErr Replace']))
            smart_replace_dict[cur_file].append(float(row['MAE Mean SReplace']))
            smart_replace_stderr_dict[cur_file].append(
                float(row['MAE StdErr Sreplace']))
            coverage_dict[cur_file].append(float(row['Coverage']))

# Create color cycle.
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']


i = 0 # i tracks the color.
# Produce the plot for Figure 8
for key in normal_dict.keys():
    plt.plot(coverage_dict[key], normal_dict[key],
             label=label_dict[key], color=color_cycle[i])
    plt.fill_between(
        coverage_dict[key],
        np.array(normal_dict[key])+np.array(normal_stderr_dict[key]),
        np.array(normal_dict[key])-np.array(normal_stderr_dict[key]),
        alpha=0.5, color=color_cycle[i])
    i += 1
plt.legend()
plt.xlabel("Coverage")
plt.ylabel("Mean Additional Error")
plt.savefig("mturk_results_filtered/normal.pdf", bbox_inches="tight")
plt.cla()

i = 0 # For the color cycle
# Produce the plot for figure 9
for key in normal_dict.keys():
    plt.plot(coverage_dict[key], replace_dict[key],
             label=label_dict[key], color=color_cycle[i])
    plt.fill_between(
        coverage_dict[key],
        np.array(replace_dict[key])+np.array(replace_stderr_dict[key]),
        np.array(replace_dict[key])-np.array(replace_stderr_dict[key]),
        alpha=0.5, color=color_cycle[i])
    i += 1
plt.legend()
plt.xlabel("Coverage")
plt.ylabel("Replacement Mean Additional Error")
plt.savefig("mturk_results_filtered/replacement.pdf", bbox_inches="tight")
plt.cla()

# Produce the plot for figures 10/11 (depending on infile)
i = 0 # for color cycle.
for key in normal_dict.keys():
    plt.plot(
        coverage_dict[key], smart_replace_dict[key],
        label=label_dict[key], color=color_cycle[i])
    plt.fill_between(
        coverage_dict[key],
        np.array(smart_replace_dict[key])+np.array(
            smart_replace_stderr_dict[key]),
        np.array(smart_replace_dict[key])-np.array(
            smart_replace_stderr_dict[key]),
        alpha=0.5, color=color_cycle[i])
    i += 1
plt.legend()
plt.xlabel("Coverage")
plt.ylabel("Replacement Mean Additional Error")
plt.savefig("mturk_results_filtered/smartreplacement.pdf", bbox_inches="tight")
plt.cla()
