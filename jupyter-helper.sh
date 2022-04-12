#!/bin/bash

#echo "jupyter init..."
source /sps/lsst/users/campagne/anaconda3/etc/profile.d/conda.sh

conda activate JaxTutos


exec python -m ipykernel_launcher "$@"
