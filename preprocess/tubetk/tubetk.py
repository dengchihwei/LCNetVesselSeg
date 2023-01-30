# -*- coding = utf-8 -*-
# @File Name : tubetk.py
# @Date : 5/17/22 2:01 AM
# @Author : zhiweideng
# @E-mail : zhiweide@usc.edu


import os
import itk
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from itk import TubeTK as ttk


# reader types
pixel_type, dimension = itk.F, 3
image_type = itk.Image[pixel_type, dimension]
mra_reader_type = itk.ImageFileReader[image_type]
tre_reader_type = itk.SpatialObjectReader[dimension]
tre2image_filter_type = ttk.ConvertTubesToImage[image_type]
# parent path
parent_path = '/ifs/loni/faculty/shi/spectrum/zdeng/MSA_Data/VesselLearning/Datasets/TubeTK'


# change .tr file to npy and nifti image
def gen_tube_label(subject_folder):
    mra_path = os.path.join(parent_path, subject_folder, 'MRA', subject_folder.replace('-', '') + '-MRA.mha')
    tre_path = os.path.join(parent_path, subject_folder, 'AuxillaryData', 'VascularNetwork.tre')

    # read tre file
    tre_file_reader = tre_reader_type.New()
    tre_file_reader.SetFileName(tre_path)
    tre_file_reader.Update()
    trees = tre_file_reader.GetGroup()

    # need mra image to change to image
    mra_file_reader = mra_reader_type.New()
    mra_file_reader.SetFileName(mra_path)
    mra_file_reader.Update()
    mra_image = mra_file_reader.GetOutput()
    mra_output_path = os.path.join(parent_path, subject_folder, '{}_MRA.mha'.format(subject_folder))
    itk.imwrite(mra_image, mra_output_path)

    # transform to binary image
    tre2image_filter = tre2image_filter_type.New()
    tre2image_filter.SetUseRadius(True)
    tre2image_filter.SetTemplateImage(mra_image)
    tre2image_filter.SetInput(trees)
    tre2image_filter.Update()

    # get output
    output_image = tre2image_filter.GetOutput()
    tre_output_path = os.path.join(parent_path, subject_folder, '{}_LABEL.mha'.format(subject_folder))
    itk.imwrite(output_image, tre_output_path)

    output_array = itk.GetArrayFromImage(output_image)
    output_path = os.path.join(parent_path, subject_folder, '{}_LABEL.npy'.format(subject_folder))
    np.save(output_path, output_array)
    print('Saved binary result ground truth mask at', output_path)


# command print helper function
def command_iteration(method):
    print("{0:3} = {1:10.5f} : {2}".format(method.GetOptimizerIteration(),
                                           method.GetMetricValue(),
                                           method.GetOptimizerPosition()))


# generate masks for MRA images with skull-striped T1 images
def gen_mask(subject_folder):
    fixed_file = os.path.join(parent_path, subject_folder, 'MRA', subject_folder.replace('-', '') + '-MRA.mha')
    moving_file = os.path.join(parent_path, subject_folder, 'AuxillaryData', 'SkullStripped-T1-Flash.mha')
    fixed_image = sitk.ReadImage(fixed_file, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(moving_file, sitk.sitkFloat32)

    # define registration operator
    registrator = sitk.ImageRegistrationMethod()
    registrator.SetMetricAsMeanSquares()
    registrator.SetOptimizerAsRegularStepGradientDescent(4.0, .01, 200)
    registrator.SetInitialTransform(sitk.TranslationTransform(fixed_image.GetDimension()))
    registrator.SetInterpolator(sitk.sitkLinear)
    registrator.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(registrator))

    out_tx = registrator.Execute(fixed_image, moving_image)
    print('-------')
    print(out_tx)
    print('Optimizer stop condition: {0}'.format(registrator.GetOptimizerStopConditionDescription()))
    print('Iteration: {0}'.format(registrator.GetOptimizerIteration()))
    print('Metric value: {0}'.format(registrator.GetMetricValue()))

    # save transformation file
    transform_file = os.path.join(parent_path, subject_folder, '{}_TRANSFORM.txt'.format(subject_folder))
    sitk.WriteTransform(out_tx, transform_file)

    # resample the mask image
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(out_tx)

    # output the image to .npy numpy array and .mha file
    output_image_file = os.path.join(parent_path, subject_folder, '{}_MASK.mha'.format(subject_folder))
    output_numpy_file = os.path.join(parent_path, subject_folder, '{}_MASK.npy'.format(subject_folder))
    out = resampler.Execute(moving_image)
    mask_data = (sitk.GetArrayFromImage(out) > 0.0).astype(float)
    registered_mask = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
    sitk.WriteImage(registered_mask, output_image_file)
    np.save(output_numpy_file, mask_data)


if __name__ == '__main__':
    subject_folders = sorted(os.listdir(parent_path))
    for folder in tqdm(subject_folders):
        modalities = os.listdir(os.path.join(parent_path, folder))
        if 'AuxillaryData' in modalities:
            gen_tube_label(folder)
            # gen_mask(folder)
