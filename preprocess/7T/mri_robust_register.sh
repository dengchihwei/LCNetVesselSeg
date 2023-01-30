#!/bin/bash

moving_file=$1
target_file=$2
transform_file=$3
options=${@:4}

export FREESURFER_HOME=/usr/local/freesurfer-6
source $FREESURFER_HOME/SetUpFreeSurfer.sh

# co-register using freesurfer
$FREESURFER_HOME/bin/mri_robust_register --mov $moving_file --dst $target_file --lta $transform_file --satit $options
