# Russ notes
using 4.5.4
conda create -n jigsaw python
conda activate jigsaw
conda install -c conda-forge gdcm 
pip install -r requirements.txt



# JigsawPuzzlePytorch
Pytorch implementation of the paper ["Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles"](https://arxiv.org/abs/1603.09246) by Mehdi Noroozi [GitHub](https://github.com/MehdiNoroozi/JigsawPuzzleSolver)

**Partially tested**
**Performances Coming Soon**

# Dependencies
- Tested with Python 2.7
- [Pytorch](http://pytorch.org/) v0.3
- [Tensorflow](https://www.tensorflow.org/) is used for logging. 
  Remove the Logger all scripts if tensorflow is missing

# Train the JigsawPuzzleSolver
## Setup Loader
Two DataLoader are provided:
- ImageLoader: per each iteration it loads data in image format (jpg,png ,...)
    - *Dataset/JigsawImageLoader.py* uses PyTorch DataLoader and iterator
    - *Dataset/ImageDataLoader.py* custom implementation.

The default loader is *JigsawImageLoader.py*. *ImageDataLoader.py* is slightly faster when using single core.

The images can be preprocessed using *_produce_small_data.py_* which resize the image to 256, keeping the aspect ratio, and crops a patch of size 255x255 in the center.

## Run Training
Fill the path information in *run_jigsaw_training.sh*. 
IMAGENET_FOLD needs to point to the folder containing *ILSVRC2012_img_train*.

```
./run_jigsaw_training.sh [GPU_ID]
```
or call the python script
```
python JigsawTrain.py [*path_to_imagenet*] --checkpoint [*path_checkpoints_and_logs*] --gpu [*GPU_ID*] --batch [*batch_size*]
```
By default the network uses 1000 permutations with maximum hamming distance selected using *select_permutations.py*.

To change the file name loaded for the permutations, open the file *JigsawLoader.py* and change the permutation file in the method *retrive_permutations*

# Details:
- The input of the network should be 64x64, but I need to resize to 75x75,
  otherwise the output of conv5 is 2x2 instead of 3x3 like the official architecture
- Jigsaw trained using the approach of the paper: SGD, LRN layers, 70 epochs
- Implemented *shortcuts*: spatial jittering, normalize each patch indipendently, color jittering, 30% black&white image
- The LRN layer crushes with a PyTorch version older than 0.3

# ToDo
- TensorboardX
- LMDB DataLoader

## Original Network
```python

Network(
  (conv): Sequential(
    (conv1_s1): Conv2d(3, 96, kernel_size=(11, 11), stride=(2, 2))
    (relu1_s1): ReLU(inplace)
    (pool1_s1): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (lrn1_s1): LRN(
      (average): AvgPool3d(kernel_size=(5, 1, 1), stride=1, padding=(2, 0, 0))
    )
    (conv2_s1): Conv2d(96, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=2)
    (relu2_s1): ReLU(inplace)
    (pool2_s1): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (lrn2_s1): LRN(
      (average): AvgPool3d(kernel_size=(5, 1, 1), stride=1, padding=(2, 0, 0))
    )
    (conv3_s1): Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (relu3_s1): ReLU(inplace)
    (conv4_s1): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2)
    (relu4_s1): ReLU(inplace)
    (conv5_s1): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2)
    (relu5_s1): ReLU(inplace)
    (pool5_s1): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (fc6): Sequential(
    (fc6_s1): Linear(in_features=2304, out_features=1024, bias=True)
    (relu6_s1): ReLU(inplace)
    (drop6_s1): Dropout(p=0.5)
  )
  (fc7): Sequential(
    (fc7): Linear(in_features=9216, out_features=4096, bias=True)
    (relu7): ReLU(inplace)
    (drop7): Dropout(p=0.5)
  )
  (classifier): Sequential(
    (fc8): Linear(in_features=4096, out_features=1000, bias=True)
  )
)
```
