#!/bin/bash

input_file=$1
output_file=$2
option=${@:3}

export FREESURFER_HOME=/usr/local/freesurfer-6
source $FREESURFER_HOME/SetUpFreeSurfer.sh

# convert_function using freesurfer
$FREESURFER_HOME/bin/mri_binarize --i $input_file --o $output_file $option
