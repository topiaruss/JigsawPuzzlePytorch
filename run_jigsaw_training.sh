#!/usr/bin/env bash
IMAGENET_FOLD=~/data/PEONY-2018-06
IMAGENET_FOLD=~/data/PEONY-2018-06/patients/HCA137955/2016-04-04/
IMAGENET_FOLD='/home/pc1/fastdata/PEONY-2018-06/patients/HCA110723/ /home/pc1/fastdata/PEONY-2018-06/patients/HCA18595/ /home/pc1/fastdata/PEONY-2018-06/patients/HCA265541/ /home/pc1/fastdata/PEONY-2018-06/patients/HCA217594/'
IMAGENET_FOLD='/home/pc1/fastdata/PEONY-2018-06/patients'
GPU=1 # gpu used
CHECKPOINTS_FOLD=${2} #path_to_output_folder

CORES=4

#python JigsawTrain.py ${IMAGENET_FOLD} --checkpoint=${CHECKPOINTS_FOLD} \
#                      --classes=1000 --batch 128 --lr=0.001 --gpu=${GPU} --cores=10
#python -m pdb JigsawTrain.py ${IMAGENET_FOLD} --classes=1000 --batch 128 --lr=0.1 --cores=${CORES} --gpu=${GPU} \
#  --epochs=500
python -m pdb JigsawTrain.py --ILSparent=/home/pc1/fastdata --classes=1000 --batch 256 --lr=0.01 --cores=${CORES} --gpu=${GPU} \
  --epochs=500
