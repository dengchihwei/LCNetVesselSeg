# -*- coding = utf-8 -*-
# @File Name : lsa_metrics
# @Date : 1/9/23 11:49 AM
# @Author : zhiweideng
# @E-mail : zhiweide@usc.edu


import sys
import time
import argparse
import subprocess
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
from sklearn import metrics
import seg_metrics.seg_metrics as sg
sys.path.append('/ifs/loni/faculty/shi/spectrum/zdeng/MSA_Data/UnsupervisedVesselSeg/codes')
from datasets.datasets_3d import get_data_loader_3d


parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='ours')
parser.add_argument('--split', type=str, default='test')


def dice_score(v_threshed, gt):
    if v_threshed.size == 0:
        return np.nan
    num = (2 * v_threshed * gt).mean()
    den = v_threshed.mean() + gt.mean() + 1e-100
    return num / den


def get_best_dice_threshold(response, label, thresholds):
    best_thresh, best_dice = None, -1
    n = int(len(thresholds) / 100.0)
    for thresh in (thresholds[::n]):
        bin_response = (response >= thresh) + 0.0
        curr_dice = dice_score(bin_response, label)
        if curr_dice > best_dice:
            best_thresh = thresh
            best_dice = curr_dice
    # print("Got best dice {:.4f} at threshold {}".format(best_dice, best_thresh))
    return best_thresh


def get_metrics(arguments):
    dice_scores, hd95_scores, avd_scores, best_thresholds = [], [], [], []
    data_loader = get_data_loader_3d(data_name='LSA', split=arguments.split, batch_size=1, shuffle=False)
    label_files = data_loader.dataset.label_files

    # compute the best thresholds
    for label_file in tqdm(label_files):
        subject_id = label_file.split('/')[-2]
        response_file = 'LSA_{}_{}/LSA_{}.nii.gz'.format(arguments.method, arguments.split, subject_id)
        response_image, label_image = sitk.ReadImage(response_file), sitk.ReadImage(label_file)
        response, label = sitk.GetArrayFromImage(response_image), sitk.GetArrayFromImage(label_image)
        _, _, thresholds = metrics.roc_curve(label.reshape(-1), response.reshape(-1), pos_label=1)
        curr_best_thresh = get_best_dice_threshold(response, label, thresholds)
        best_thresholds.append(curr_best_thresh)

    final_threshold = np.mean(best_thresholds)
    print('Final Threshold is {}.'.format(final_threshold))

    # compute the dice scores
    for label_file in tqdm(label_files):
        subject_id = label_file.split('/')[-2]
        response_file = 'LSA_{}_{}/LSA_{}.nii.gz'.format(arguments.method, arguments.split, subject_id)
        binary_file = 'LSA_{}_{}/BIN_LSA_{}.nii.gz'.format(arguments.method, arguments.split, subject_id)
        response_image, label_image = sitk.ReadImage(response_file), sitk.ReadImage(label_file)
        response, label = sitk.GetArrayFromImage(response_image), sitk.GetArrayFromImage(label_image)
        bin_response = (response >= final_threshold) + 0.0
        # write the binary results
        bin_image = sitk.GetImageFromArray(bin_response)
        bin_image.CopyInformation(response_image)
        sitk.WriteImage(bin_image, binary_file)
        # get metrics
        labels = [0, 1]
        metric_results = sg.write_metrics(labels=labels[1:],
                                          gdth_path=label_file,
                                          pred_path=binary_file,
                                          metrics=['dice', 'hd95'],
                                          verbose=False)
        curr_dice = metric_results[0]['dice'][0]
        curr_hd95 = metric_results[0]['hd95'][0]
        dice_scores.append(curr_dice)
        hd95_scores.append(curr_hd95)
        # get AVD metric
        result1 = subprocess.run(['./EvaluateSegmentation', label_file, binary_file], capture_output=True, text=True)
        curr_avd = float(result1.stdout[result1.stdout.find('AVGDIS')+9:result1.stdout.find('AVGDIS')+19])
        avd_scores.append(curr_avd)

    print("Method: {}".format(arguments.method))
    print("Dice Mean: {:.5f}, Std: {:.5f}".format(np.mean(dice_scores), np.std(dice_scores)))
    print("HD95 Mean: {:.5f}, Std: {:.5f}".format(np.mean(hd95_scores), np.std(hd95_scores)))
    print("AVD Mean: {:.5f}, Std: {:.5f}".format(np.mean(avd_scores), np.std(avd_scores)))


if __name__ == '__main__':
    args = parser.parse_args()
    get_metrics(args)
