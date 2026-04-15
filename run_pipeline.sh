#!/bin/bash
cd /Users/sivakarthick/s2
source venv/bin/activate
export TF_CPP_MIN_LOG_LEVEL=2
export TF_FORCE_GPU_ALLOW_GROWTH=true
python main_pipeline.py 2>&1
