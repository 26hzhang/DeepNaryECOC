# Deep N-ary Error Correcting Output Codes

This is TensorFlow implementation for the paper "Deep N-ary Error Correcting Output Codes" (**MobiMedia 2020**): 
[https://arxiv.org/pdf/2009.10465.pdf](https://arxiv.org/pdf/2009.10465.pdf).

## Prerequisites

- python 3.x with tensorflow (`1.8.0~1.13.1`), keras, numpy, sklearn, scipy, matplotlib, tqdm.

## Datasets
### Text Datasets
- TREC: Question-type classification dataset (6-classes), [[link]](http://cogcomp.org/Data/QA/QC/).
- STS-5: Stanford Sentiment Treebank dataset (5-classes), [[link]](https://nlp.stanford.edu/sentiment/index.html).

> Convert file format to `UTF-8` on Mac OS X: `iconv -f <other_format> -t utf-8 file > new_file`  
> Convert file format to `UTF-8` on Ubuntu Linux: `iconv -f <other_format> -t utf-8 file -o new_file`

### Image Datasets
- MNIST: Handwritten digit dataset (10-classes), [[link]](http://yann.lecun.com/exdb/mnist/).
- CIFAR: Real-world image dataset (10/100-classes), [[link]](https://www.cs.toronto.edu/~kriz/cifar.html).
- FLOWER-102: 102 Category Flower Dataset (102-classes), [[link]](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
  or [[download]](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz). (FLOWER-102 utilizes the 
  pretrained AlexNet, which can be downloaded here: [[kratzert/finetune_alexnet_with_tensorflow]](
  https://github.com/kratzert/finetune_alexnet_with_tensorflow)).
  
## Quick Start
Take the MNIST dataset as an example. Training the model for MNIST dataset without sharing base learners' parameters:
```shell
python train_mnist.py --gpu_idx 0 \  # specify the gpu index
                      --training True \  # specify the mode (training or testing)
                      --num_meta_class 3 \  # specify the number of meta classes
                      --num_classifier 60 \  # specify the number of base learners
```
Training the model for MNIST dataset with full sharing base learners' encoder parameters (except for the classifier head):
```shell
python train_mnist_full.py --gpu_idx 0 \  # specify the gpu index
                           --training True \  # specify the mode (training or testing)
                           --num_meta_class 3 \  # specify the number of meta classes
                           --num_classifier 60 \  # specify the number of base learners
```

## Citation
If you feel this project helpful to your research, please cite our work.
```text
@article{zhang2020deep,
  title={Deep N-ary Error Correcting Output Codes},
  author={Zhang, Hao and Zhou, Joey Tianyi and Wang, Tianying and Tsang, Ivor W and Goh, Rick Siow Mong},
  journal={arXiv preprint arXiv:2009.10465},
  year={2020},
  url={https://arxiv.org/pdf/2009.10465.pdf}
}
```
and
```text
@article{zhou2019n,
  title={N-ary decomposition for multi-class classification},
  author={Zhou, Joey Tianyi and Tsang, Ivor W and Ho, Shen-Shyang and M{\"u}ller, Klaus-Robert},
  journal={Machine Learning},
  volume={108},
  number={5},
  pages={809--830},
  year={2019},
  publisher={Springer}
}
```
