IMAGENET_FOLD=/home/pc1/fastdata/

GPU=1 # gpu used

#python JigsawTrain.py ${IMAGENET_FOLD} --checkpoint=${CHECKPOINTS_FOLD} \
#                      --classes=1000 --batch 128 --lr=0.001 --gpu=${GPU} --cores=10
python -m pdb JigsawTrain.py ${IMAGENET_FOLD} --classes=1000 --batch 256 --lr=0.01 --gpu=${GPU} --cores=4
