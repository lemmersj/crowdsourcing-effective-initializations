"""Generate the scoring tsv by executing the tracker and scoring functions.

Based on the DaSiamRPN code available at https://github.com/foolwood/DaSiamRPN.

Typical usage:
    python generate_scored_mturk_tsv.py --out_dir [] --guess_iou_weights []
    --mturk_tsv [] --description_tsv []
"""
# This was written with a different configuration than I typically use.
# And I have no qualms about disabling no-member.
# pylint: disable=bad-indentation, no-member

from __future__ import print_function
import argparse
import cv2
import torch
import json
import numpy as np
from os import makedirs
from os.path import realpath, dirname, join, exists

from torchvision import models
from net import SiamRPNotb
from run_SiamRPN import SiamRPN_init, SiamRPN_track
from utils import rect_2_cxy_wh, cxy_wh_2_rect
import utils_2
import csv
from IPython import embed
import random

parser = argparse.ArgumentParser(description='Generate scoring tsv.')
parser.add_argument('--out_dir', type=str)
parser.add_argument('--guess_iou_weights', type=str)
parser.add_argument('--mturk_tsv', type=str)
parser.add_argument('--description_tsv', type=str)

args = parser.parse_args()

def track_video(model, video, iou_net, start_bbox, worker_id):
    """Calculates scores and IoUs on a video given a starting bbox.

    Args:
        model: The DaSiamRPN model.
        video: The video.
        iou_net: The model which estimates the first-frame iou.
        start_bbox: The initialization bbox.
        worker_id: The randomID of the worker.

    Returns:
        A dictionary containing the output of various scoring functions, along
        with the gold-standard and initialization IoUs.
    """
    # Contains all of the scores which are written to the file
    return_dict = {}
    iou_outputs_mean_list = []
    iou_outputs_argmax_list = []

    # Images are pulled from the dictionary
    image_files, gt = video['image_files'], video['gt']

    # Split the ground truth bbox (in "rect" form) to pos and sz
    target_pos = torch.tensor((gt[0][0], gt[0][1]))
    target_sz = torch.tensor((gt[0][2], gt[0][3]))

    # generate perturbed bboxes
    annot_start_pos = start_bbox[:2].astype('double')
    annot_start_sz = start_bbox[2:].astype('double')

    # Do a pass with the ground truth
    regions = []
    for f, image_file in enumerate(image_files):
        im = cv2.imread(image_file)  # TODO: batch load
        if f == 0:  # init
            # the tracker wants cxy w h
            target_pos, target_sz = rect_2_cxy_wh(gt[f])
            # initialize the tracker.
            state = SiamRPN_init(im, target_pos, target_sz, model)

            # but we use rect representation for iou calculations.
            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            regions.append(location)
        elif f > 0:  # track
            state = SiamRPN_track(state, im)  # track
            # we do all of our calculations in "rect" which is tlx tly w h
            location = cxy_wh_2_rect(
                state['target_pos']+1, state['target_sz'])
            regions.append(location)

    gt_regions = regions
    # both regions and gt are "rect"
    return_dict['GT IoU'] = utils_2.calc_vid_iou(regions, gt)
    where_zero = torch.where(return_dict['GT IoU'] <= 1e-6)

    if len(where_zero[0]) == 0:
        last_frame = len(return_dict['GT IoU'])-1
    else:
        last_frame = where_zero[0].min().item()
    if last_frame < 0:
        embed()
    return_dict['last_frame'] = last_frame+1

    # and do a pass with the annotator's seed
    regions = []
    siamese_scores = []
    for f, image_file in enumerate(image_files):
        im = cv2.imread(image_file)  # TODO: batch load
        if f == 0:  # init
            # convert from rect to cxy
            cxy_annot_start_pos = annot_start_pos + annot_start_sz/2.
            # initialize tracker
            state = SiamRPN_init(im, cxy_annot_start_pos, annot_start_sz, model)
            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            regions.append(location)
            siamese_scores.append(1.)
        elif f > 0:  # tracking
            state = SiamRPN_track(state, im)  # track
            location = cxy_wh_2_rect(
                state['target_pos']+1, state['target_sz'])
            regions.append(location)
            siamese_scores.append(state['score'])

        # Produce the masked images for the iou regressor
        masked_image = utils_2.add_fourth_channel(
            im, torch.tensor(regions[-1][:2]), torch.tensor(regions[-1][2:]))
        iou_outputs = iou_net(masked_image.cuda().unsqueeze(0))
        iou_outputs = torch.softmax(iou_outputs[:, :10], dim=1)
        iou_output_mean = (
            torch.arange(10) * iou_outputs.squeeze().cpu()).sum().item()
        iou_outputs_mean_list.append(iou_output_mean)
        iou_outputs_argmax_list.append(iou_outputs.argmax().item())

    rand_regions = regions
    return_dict['Perturbed IoU'] = utils_2.calc_vid_iou(regions, gt)
    return_dict['name'] = video['name']

    vid_dir = join(args.out_dir, worker_id, video['name'])
    makedirs(vid_dir)
    with open(join(vid_dir, "summary.csv"), 'w') as outfile:
        outfile.write("Perturbed,Gold Standard\n")

    # draw the video
    for f, image_file in enumerate(image_files):
        im = cv2.imread(image_file)  # TODO: batch load
        # tlx tly w h
        with open(join(vid_dir, "summary.csv"), 'a') as outfile:
            outfile.write(str(return_dict['Perturbed IoU'][f].item())+","+
                          str(return_dict['GT IoU'][f].item())+"\n")
        cv2.rectangle(im, (int(rand_regions[f][0]), int(rand_regions[f][1])),
                      (int(rand_regions[f][0] + rand_regions[f][2]),
                       int(rand_regions[f][1] + rand_regions[f][3])),
                      (0, 0, 255), 3)
        cv2.rectangle(im, (int(gt_regions[f][0]), int(gt_regions[f][1])),
                      (int(gt_regions[f][0] + gt_regions[f][2]),
                       int(gt_regions[f][1] + gt_regions[f][3])),
                      (0, 255, 255), 3)
        cv2.rectangle(im, (int(gt[f][0]), int(gt[f][1])),
                      (int(gt[f][0] + gt[f][2]), int(gt[f][1] + gt[f][3])),
                      (0, 255, 0), 3)
        cv2.imwrite(join(vid_dir, str(f)+".jpg"), im)
        if f == 0:
            im = cv2.imread(image_file)  # TODO: batch load
            cv2.rectangle(im, (int(rand_regions[f][0]),
                               int(rand_regions[f][1])),
                          (int(rand_regions[f][0] + rand_regions[f][2]),
                           int(rand_regions[f][1] + rand_regions[f][3])),
                          (0, 0, 255), 3)
            cv2.imwrite(join(vid_dir, "first.jpg"), im)

    # Using the last frame of the randomly generated start, track in reverse
    start = True
    while f >= 0:
        im = cv2.imread(image_files[f])  # TODO: batch load
        if start:
            # init tracker
            state = SiamRPN_init(im, state['target_pos'], state['target_sz'],
                                 model)
            start = False
        else:  # tracking
            state = SiamRPN_track(state, im)  # track

        f -= 1

    # Change to tlx tly w h from center
    reverse_location = cxy_wh_2_rect(state['target_pos']+1, state['target_sz'])

    # and calculate the IoU
    return_dict['reverse_score_all'] = utils_2.calc_iou(
        torch.cat((torch.tensor(reverse_location[:2]).float(),
                   torch.tensor(
                       reverse_location[2:]).float())).unsqueeze(0).float(),
        torch.cat((torch.tensor(annot_start_pos),
                   torch.tensor(annot_start_sz))).unsqueeze(0).float()).item()

    # Now do it from the last frame where the gt start has overlap.
    start = True
    f = last_frame
    while f >= 0:
        im = cv2.imread(image_files[f])
        if start:  # init
            # remember to undo the +1 used when storing regions.
            new_start_pos, new_start_sz = rect_2_cxy_wh(
                regions[f] - np.array([1, 1, 0, 0]))
            # init tracker
            state = SiamRPN_init(im, new_start_pos, new_start_sz, model)
            start = False
        else:  # tracking
            state = SiamRPN_track(state, im)  # track

        f -= 1

    # Change to tlx tly w h from center
    reverse_location = cxy_wh_2_rect(state['target_pos']+1, state['target_sz'])

    # and calculate the IoU
    return_dict['reverse_score_trunc'] = utils_2.calc_iou(
        torch.cat((torch.tensor(reverse_location[:2]).float(),
                   torch.tensor(
                       reverse_location[2:]).float())).unsqueeze(0).float(),
        torch.cat((torch.tensor(annot_start_pos),
                   torch.tensor(annot_start_sz))).unsqueeze(0).float()).item()
    return_dict['iou_mean_first'] = iou_outputs_mean_list[0]
    return_dict['iou_mean_all'] = np.array(iou_outputs_mean_list).mean()
    return_dict['iou_mean_all_trunc'] = np.array(
        iou_outputs_mean_list)[:last_frame+1].mean()
    return_dict['iou_mean_last'] = np.array(iou_outputs_mean_list)[-1]
    return_dict['iou_mean_last_trunc'] = np.array(
        iou_outputs_mean_list)[last_frame]
    return_dict['iou_argmax_all'] = np.array(iou_outputs_argmax_list).mean()
    return_dict['iou_argmax_all_trunc'] = np.array(
        iou_outputs_argmax_list)[:last_frame+1].mean()
    return_dict['iou_argmax_last'] = np.array(iou_outputs_argmax_list)[-1]
    return_dict['iou_argmax_last_trunc'] = np.array(
        iou_outputs_argmax_list)[last_frame]
    return_dict['vid_length'] = return_dict['Perturbed IoU'].shape[0]
    return_dict['iou_true'] = return_dict['Perturbed IoU'][0].item()
    return_dict['tracker_confidence'] = np.array(siamese_scores).mean()
    return_dict['tracker_confidence_trunc'] = np.array(
        siamese_scores)[:last_frame+1].mean()
    return_dict['Perturbed IoU_trunc'] = return_dict['Perturbed IoU'][
        :last_frame+1].mean().item()
    return_dict['Perturbed IoU'] = return_dict['Perturbed IoU'].mean().item()
    return_dict['GT IoU_trunc'] = return_dict['GT IoU'][
        :last_frame+1].mean().item()
    return_dict['GT IoU'] = return_dict['GT IoU'].mean().item()
    return_dict['additional_error'] = np.max(
        (return_dict['GT IoU']-return_dict['Perturbed IoU'], 0))
    return_dict['additional_error_trunc'] = np.max(
        (return_dict['GT IoU_trunc']-return_dict['Perturbed IoU_trunc'],
         0)).mean()
    return_dict['random'] = random.random()

    return return_dict

def load_dataset(dataset):
    """Loads the OTB dataset.

    Args:
        dataset: Should always be OTB2015.

    Returns:
        A dictionary where the key corresponds to the video name, and consists
        of a list of gold-standard bounding boxes and image files.
    """

    base_path = join(realpath(dirname(__file__)), 'data', dataset)
    if not exists(base_path):
        print("Please download OTB dataset into `data` folder!")
        exit()
    json_path = join(realpath(dirname(__file__)), 'data', dataset + '.json')
    info = json.load(open(json_path, 'r'))
    for v in info.keys():
        path_name = info[v]['name']
        info[v]['image_files'] = [
            join(base_path, path_name, 'img', im_f) for im_f in\
            info[v]['image_files']]
        # tracker is zero index
        info[v]['gt'] = np.array(info[v]['gt_rect'])-[1, 1, 0, 0]
        info[v]['name'] = v
    return info


def main():
    """The main function.

    Initializes the various models, reads the annotations, and creates
    the summary files.
    """
    makedirs(args.out_dir)

    iou_network = models.resnet18(pretrained=True).cuda()
    iou_network.conv1.weight = torch.nn.Parameter(
        torch.cat((iou_network.conv1.weight.detach(),
                   torch.rand(64, 1, 7, 7).cuda()), dim=1))
    iou_network.load_state_dict(torch.load(args.guess_iou_weights))

    net = SiamRPNotb()
    net.load_state_dict(torch.load(join(realpath(dirname(__file__)),
                                        'SiamRPNOTB.model')))
    net.eval().cuda()

    dataset = load_dataset('OTB2015')

    first_row = True

    to_upper_vid_dict = {}
    for key in dataset.keys():
        to_upper_vid_dict[key.upper()] = key

    description_dict = {}
    with open(args.description_tsv) as description_tsv:
        description_reader = csv.DictReader(description_tsv, delimiter="\t")
        for row in description_reader:
            description_dict[row['Name'].upper()] = row['Description']
    with open(args.mturk_tsv, 'r') as mturk_tsv:
        mturk_reader = csv.DictReader(mturk_tsv, delimiter="\t")
        for row in mturk_reader:
            with torch.no_grad():
                init_bbox = np.array([row['left'],
                                      row['top'],
                                      row['width'],
                                      row['height']])
                dataset_key = to_upper_vid_dict[row['label'].upper()]
                return_dict = track_video(
                    net, dataset[dataset_key],
                    iou_network, init_bbox, row['RandomID'])
            if first_row:
                first_row = False
                mturk_key_list = row.keys()
                score_key_list = return_dict.keys()
                with open(args.out_dir+"/summary.tsv", "w")\
                        as outfile:
                    outfile.write('name\t')
                    outfile.write('description\t')
                    for key in mturk_key_list:
                        outfile.write(key)
                        outfile.write("\t")
                    for key in score_key_list:
                        if key == 'name':
                            continue
                        outfile.write(key)
                        outfile.write("\t")
                    outfile.write("\n")

            with open(args.out_dir+"/summary.tsv", "a")\
                    as outfile:
                        outfile.write(return_dict['name']+"\t")
                        outfile.write(
                            description_dict[return_dict['name'].upper()]+"\t")
                        for key in mturk_key_list:
                            outfile.write(str(row[key]))
                            outfile.write("\t")
                        for key in score_key_list:
                            if key == 'name':
                                continue
                            outfile.write(str(return_dict[key]))
                            outfile.write("\t")
                        outfile.write("\n")

if __name__ == '__main__':
    main()
