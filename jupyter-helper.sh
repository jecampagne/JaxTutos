#!/bin/bash

#echo "jupyter init..."
source /sps/lsst/users/campagne/anaconda3/etc/profile.d/conda.sh

conda activate JaxTutos

export XLA_PYTHON_CLIENT_PREALLOCATE=false

exec python -m ipykernel_launcher "$@"
