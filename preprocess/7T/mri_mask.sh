#!/bin/bash

input_file=$1
mask_file=$2
output_file=$3
option=${@:4}

export FREESURFER_HOME=/usr/local/freesurfer-6
source $FREESURFER_HOME/SetUpFreeSurfer.sh

# convert_function using freesurfer
$FREESURFER_HOME/bin/mri_mask $option $input_file $mask_file $output_file
