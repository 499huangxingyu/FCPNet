#!/bin/bash

cd 'your train files'

nohup python -u Dense_train.py > FCPNet.out 2>&1 &

echo "========================================================"
echo "The file is running，and the outputs will saved in out"
echo "========================================================"

tail -f FCPNet.out