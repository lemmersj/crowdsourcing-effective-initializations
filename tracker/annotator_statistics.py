"""Calculate statistics related to annotator performance.

Reads in the output of generate_scored_mturk_tsv.py (a tsv file) and produces
various statistics that are used in the paper.

Typical usage example:

    python annotator_statistics.py summary.tsv
"""
from __future__ import print_function
import sys
import csv
import numpy as np

# The number of annotations performed by each annotator.
annotator_dict = {}
# The number of incorrect (iou < 0.5) annotations performed by each annotator.
bad_annotator_dict = {}
# Total number of incorrect annotations.
bad_iou_count = 0
# Total number of annotations.
total_count = 0

# Read all the annotations.
with open(sys.argv[1]) as tsv_infile:
  tsv_file = csv.DictReader(tsv_infile, delimiter="\t")
  for row in tsv_file:
    # Increment the total number of annotations.
    total_count += 1

    # If we haven't seen this annotator before, add him to both dicts.
    if row['RandomID'] not in annotator_dict.keys():
      annotator_dict[row['RandomID']] = 0
      bad_annotator_dict[row['RandomID']] = 0

    # Increment the number of annotations by this annotator.
    annotator_dict[row['RandomID']] += 1

    # If applicable, increment the number of incorrect annotations
    # for this annotator.
    if float(row['iou_true']) < 0.5:
      bad_annotator_dict[row['RandomID']] += 1
      bad_iou_count += 1

# Print the first bit of information: how many annotations are incorrect?
print("A total of ", bad_iou_count, " of ", total_count,
      " annotations have a high iou.")

# How many annotations were performed by each annotator.
all_annotation_counts = []

# How many annotators are good (less than 15% of annotations are invalid)
good_annotator_count = 0

# How many annotators are bad (greater than 15% of annotations are invalid)
bad_annotator_count = 0

good_annotator_good_result = 0 # for p(good_result| good_annotator)
good_annotator_bad_result = 0 # for p(bad_result | good_annotator)
good_annotator_total_results = 0 # for both above
bad_annotator_good_result = 0 # for p(good_result | bad_annotator)
bad_annotator_bad_result = 0 # for p(bad_result | bad_annotator)
bad_annotator_total_results = 0 # for both above

# print individual annotator statistics, and update above variables.
for key in annotator_dict:
  # Save total number of counts
  all_annotation_counts.append(annotator_dict[key])

  # Print individual annotator statistics.
  print("annotator ", key, " had ", bad_annotator_dict[key],
        " bad annotations out of ", str(annotator_dict[key]),
        " total annotations (",
        float(100*bad_annotator_dict[key])/annotator_dict[key], "%)")

  # Update probability variables.
  # if > 15 percent were incorrect, the annotator is bad.
  if float(100*bad_annotator_dict[key])/annotator_dict[key] > 15:
    # Update the various counters.
    bad_annotator_count += 1
    bad_annotator_good_result += annotator_dict[key] - bad_annotator_dict[key]
    bad_annotator_bad_result += bad_annotator_dict[key]
    bad_annotator_total_results += annotator_dict[key]
  else:
    # Update the various counters.
    good_annotator_count += 1
    good_annotator_good_result += annotator_dict[key] - bad_annotator_dict[key]
    good_annotator_bad_result += bad_annotator_dict[key]
    good_annotator_total_results += annotator_dict[key]

# Calculate probabilities.
p_good_measurement = 1-(float(bad_iou_count)/total_count)
p_good_annotator = good_annotator_count /\
        float(good_annotator_count+bad_annotator_count)
p_bad_annotator = bad_annotator_count /\
        float(good_annotator_count+bad_annotator_count)
p_good_given_good_annotator = good_annotator_good_result /\
        float(good_annotator_total_results)
p_bad_given_good_annotator = good_annotator_bad_result /\
        float(good_annotator_total_results)
p_good_given_bad_annotator = bad_annotator_good_result /\
        float(bad_annotator_total_results)
p_bad_given_bad_annotator = bad_annotator_bad_result /\
        float(bad_annotator_total_results)

print("p(correct) = ", p_good_measurement)
print("p(correct|good_annotator) =", p_good_given_good_annotator)
print("p(correct|bad_annotator) =", p_good_given_bad_annotator)
print("p(incorrect|good_annotator) =", p_bad_given_good_annotator)
print("p(incorrect|bad_annotator) =", p_bad_given_bad_annotator)
print("p(good_annotator) =", p_good_annotator)
print("p(bad_annotator) =", p_bad_annotator)

# Joint likelihood calculated using chain rule.
print("L(good_annot, good_1, good_2) [method 1] = ",
      p_good_given_good_annotator*p_good_given_good_annotator*p_good_annotator)
# Calculating p(good_annotator | good_annotation, good_annotation)
p_good_annotator_given_two = p_good_given_good_annotator*\
        p_good_given_good_annotator*p_good_annotator /\
        (p_good_measurement*p_good_measurement)
print("L(good_annotator | good_1, good_2) = ", p_good_annotator_given_two)
# And reversing it to sanity check.
print("L(good_annot, good_1, good_2) [method 2] = ",
      p_good_annotator_given_two*p_good_measurement*p_good_measurement)

# Same process for bad annotator given two good annotations.
print("L(bad_annot, good_1, good_2) [method 1] = ",
      p_good_given_bad_annotator*p_good_given_bad_annotator*p_bad_annotator)
p_bad_annotator_given_two = p_good_given_bad_annotator*\
        p_good_given_bad_annotator*p_bad_annotator /\
        (p_good_measurement*p_good_measurement)
print("L(bad_annotator | good_1, good_2) = ", p_bad_annotator_given_two)
print("L(bad_annot, good_1, good_2) [method 2] = ", p_bad_annotator_given_two*\
      p_good_measurement*p_good_measurement)

# Simple bayes rule.
p_bad_annotator_given_one = p_good_given_bad_annotator*p_bad_annotator/\
        p_good_measurement
print("L(bad_annotator | good_1) = ", p_bad_annotator_given_one)
p_good_annotator_given_one = p_good_given_good_annotator*p_good_annotator/\
        p_good_measurement
print("L(good_annotator | good_1) = ", p_good_annotator_given_one)

# Convert the likelihoods to probabilities
p_bad_annotator_given_two_normalized = p_bad_annotator_given_two/\
        (p_bad_annotator_given_two+p_good_annotator_given_two)
print("p(bad_annot | good_1, good_2) = ", p_bad_annotator_given_two_normalized)
p_good_annotator_given_two_normalized = p_good_annotator_given_two/\
        (p_bad_annotator_given_two+p_good_annotator_given_two)
print("p(good_annot | good_1, good_2) = ",
      p_good_annotator_given_two_normalized)
p_bad_annotator_given_one_normalized = p_bad_annotator_given_one/\
        (p_bad_annotator_given_one+p_good_annotator_given_one)
print("p(bad_annot | good_1) = ", p_bad_annotator_given_one_normalized)
p_good_annotator_given_one_normalized = p_good_annotator_given_one/\
        (p_bad_annotator_given_one+p_good_annotator_given_one)
print("p(good_annot | good_1) = ", p_good_annotator_given_one_normalized)

# Annotator statistics.
print(str(len(annotator_dict.keys())) + " annotators")
print("each provided (mean) ", np.array(all_annotation_counts).mean(),
      " annotations")
print("With a standard deviation of ", np.array(all_annotation_counts).std())
