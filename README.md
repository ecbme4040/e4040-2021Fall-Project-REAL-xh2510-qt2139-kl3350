# Re-implement the work in paper: Residual Attention Networks for Image Classification
Xin Huang (UNI: xh2510), Qimeng Tao (UNI: qt2139), Kangrui Li (UNI: kl3350)
##  Introduction
In recent years, nature of extracted attention has been studied in previous work and formulating attention drift is found to enhance image classification. Another advanced technology is the proposal of 'Deep Residual Network', which solved problems in deep neural networks. Based on these two techniques, the authors of this paper proposed a combination of attention module and residual network and this Attention Residual Network achieved great performance in image classification tasks.

In our implementation, based on the original architecture, we modified some units in it and generated our own attention residual network. We evaluated the performance of our design on both CIFAR-10 and CIFAR-100 dataset and obtained nearly the same results in the original paper. We also tested the performance of shortcut connection by replacing it with convolution layers. From the results, it truns out applying shortcut connection increases performance of deep network.
##  Dataset
In our project, we used CIFAR-10 and CIFAR-100 to train and test our model. The CIFAR-10 dataset consists of 60000 32x32 color images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. 

CIFAR-100 dataset is just like the CIFAR-10, except it has 100 classes containing 600 images each. There are 500 training images and 100 testing images per class. The 100 classes in the CIFAR-100 are grouped into 20 superclasses. Each image comes with a "fine" label (the class to which it belongs) and a "coarse" label (the superclass to which it belongs)

##  Results
Our network structure of Attention 56 and Attention 92
<img width="634" alt="table1" src="https://user-images.githubusercontent.com/90971979/146977436-0e3f3a8e-2d0c-43a0-a813-08e621ee2745.png">

First of all, we use the data set on CIFAR-10, first train the model on the network of Attention 56. The results are shown in the table below.
| Network Structure | Error | Reference Error | Traning Times |
| ----------------- | ----- | --------------- | ------------- |
| With Shortcut     | 6.71% | 5.52% | 107s / epochs(bath size = 64) |
| Without Shortcut  | 6.68% | 5.52% | 98s  / epochs(bath size = 64) |

Then we use the data set on CIFAR-10, train the model on the network of Attention 92. The results are shown in the table below.
| Network Structure | Error | Reference Error | Traning Times |
| ----------------- | ----- | --------------- | ------------- |
| With Shortcut     | 5.91% | 4.99% | 109s / epochs(bath size = 64) |
| Without Shortcut  | 6.12% | 4.99% | 99s  / epochs(bath size = 64) |

We also implemented the training of Attention56 and Attention92 on CIFAR-100. The results are shown in the table below.
Attention 56 on CIFAR-100
| Network Structure | Error | Reference Error | Traning Times |
| ----------------- | ----- | --------------- | ------------- |
| With Shortcut     | 32.24% | Not provided | 172s / epochs(bath size = 64) |
| Without Shortcut  | 37.98% | Not provided | 172s / epochs(bath size = 64) |

Attention 92 on CIFAR-100
| Network Structure | Error | Reference Error | Traning Times |
| ----------------- | ----- | --------------- | ------------- |
| With Shortcut     | 25.8% | 20.71% | 424s / epochs(bath size = 64) |
| Without Shortcut  | 30.3% | 20.71% | 424  / epochs(bath size = 64) |

Finally, we discussed the impact of noise on accuracy, and the results are shown in the following table.
| Noise | ResNet 164 | Attention 56 | Attention 92 |
| ----- | ---------- | ------------ | ------------ |
| 10%   | 5.93% | 9.46%  | 7.35% |
| 30%   | 6.61% | 10.43% | 8.21% |
| 50%   | 8.35% | 13.10% | 10.31%|
| 70%   | 17.21%| 21.3%  | 18.3% |

##  Environment
Python 3.8
Tensorflow-gpu 2.5
Google Cloud (NVIDIA Tesla P100)

## Attention92 model for CIFAR-10
link: https://drive.google.com/drive/folders/12_ZY38endjkQfbJTuMWoFz2vwR6pbcin?usp=sharing

##  Team Member and Contribution
| Name | UNI | Details 1 | Details 2 | Details 3 | Fraction of total contribution |
| ---- | --- | --------- | --------- | --------- | ------------------------------ |
| Xin Huang   | xh2510   | Building Model Structure | Training Model | Writing Paper | 1/3 |
| Qimeng Tao  | qt2139   | Building Model Structure | Training Model | Writing Paper | 1/3 |
| Kangrui Li  | kl3350   | Building Model Structure | Training Model | Writing Paper | 1/3 |


./  
├── ECBM_4040_Final_Project.ipynb  
├── Modual  
│   ├── Attention Module without shortcut.py  
│   ├── attention_module_with shortcut.py  
│   └── residual_unit.py  
├── Project Report.pdf  
├── README.md  
├── Training Model  
│   ├── Attention_56_Model_for_Cifar10.py  
│   ├── Attention_56_Model_for_Cifar_100.py  
│   ├── Attention_92_Model_for_Cifar_10.py  
│   ├── Attention_92_Model_for_Cifar_100.py  
│   └── noise_level.py  
└── figures  
    ├── Attention 56 acc.png  
    ├── Attention 56 loss.png  
    ├── Attention Module Architecture.png  
    ├── Example Architecture.png  
    ├── Screenshot  
    │   ├── Screenshot1.png  
    │   ├── Screenshot2.png  
    │   └── Screenshot3.png
    ├── Shortcut Connection.png  
    ├── Structure of two branches.png  
    ├── System Architecture.png  
    └── soft mask.png  

