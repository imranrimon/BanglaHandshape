#!/bin/bash
# A6 prep (login node — needs internet): download the MediaPipe hand-landmarker
# model and create the isolated mp_kp env (Tasks API, python 3.11). Once this is
# done, extraction+training run offline via a SLURM job.
set -e
source ~/miniconda3/etc/profile.d/conda.sh
cd /scratch/mh00145/BanglaHandshape

echo "=== [1/2] download hand_landmarker.task ==="
mkdir -p .kp_models
if [ -s .kp_models/hand_landmarker.task ]; then
  echo "already present ($(stat -c%s .kp_models/hand_landmarker.task) bytes)"
else
  curl -sL --retry 5 -o .kp_models/hand_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
  echo "downloaded $(stat -c%s .kp_models/hand_landmarker.task) bytes"
fi

echo "=== [2/2] create mp_kp env (python 3.11 + mediapipe Tasks API) ==="
if conda env list | grep -q '/mp_kp$'; then
  echo "mp_kp env already exists"
else
  conda create -n mp_kp python=3.11 -y
fi
conda activate mp_kp
python -m pip install --quiet --upgrade pip
python -m pip install --quiet mediapipe opencv-python-headless numpy
echo "=== mp_kp ready: $(python -c 'import mediapipe as mp; print("mediapipe", mp.__version__)') ==="
python -c "from mediapipe.tasks.python import vision; from mediapipe.tasks.python.core.base_options import BaseOptions; print('Tasks API import OK')"
echo "=== A6 PREP COMPLETE ==="
