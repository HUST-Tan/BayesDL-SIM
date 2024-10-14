 # Bayesian Deep-Learning Structured Illumination Microscopy Enables Reliable Super-Resolution Imaging with Uncertainty Quantification

**Liu Tao, Liu Jiahao, Tan Shan, Li Dong**

---

> **Abstract**: The objective of optical super-resolution (SR) imaging is to acquire reliable sub-diffraction information on bioprocesses to facilitate scientific discovery. Structured illumination microscopy (SIM) is acknowledged as the optimal modality for live-cell SR imaging. Although recent deep learning (DL) techniques have substantially advanced SIM, their transparency and reliability remain uncertain and under-explored, often resulting in unreliable SR results and biological misinterpretation. Here, we develop Bayesian deep learning (BayesDL) for SIM, which enhances the reconstruction of densely labeled structures while enabling the quantification of uncertainty of SR results. With the uncertainty, BayesDL-SIM achieves high-fidelity distribution-informed SR imaging, allowing for the communication of credibility estimates to users regarding the model outcomes. We also demonstrate that BayesDL-SIM boosts SIM reliability by identifying and preventing erroneous generalizations in various model misuse scenarios. Moreover, the BayesDL uncertainty shows versatile utilities for daily SR imaging, such as SR error estimation, data acquisition evaluation, etc. Furthermore, we demonstrate the effectiveness and superiority of BayesDL-SIM in live-cell imaging, which reliably reveals F-actin dynamics and the reorganization of the cell cytoskeleton. This work lays the foundation for the reliable implementation of DL-SIM methods in practical applications.

---



## Contents

+ Overview
+ Requirements
+ Installation
+ Quickstart (demo)
+ How to train BayesDL
+ Citation
+ License
+ Contact



## Overview

Structured illumination microscopy (SIM) is acknowledged as the optimal modality for live-cell super-resolution (SR) imaging. It typically requires nine (3 orientations × 3 phases) raw images to reconstruct an 2D SR-SIM image. Recently, deep learning (DL) techniques have widely used for end-to-end SIM reconstruction and demonstrated superior performance over traditional methods [[1](https://www.nature.com/articles/s41467-020-15784-x), [2](https://www.nature.com/articles/s41592-020-01048-5), [3](https://www.nature.com/articles/s41592-021-01155-x)].   

Although recent DL-SIM techniques have substantially advanced SIM, their transparency and reliability remain uncertain and under-explored, often resulting in unreliable SR results and biological misinterpretation. Here, we develop Bayesian deep learning (BayesDL) for SIM, which enhances the reconstruction of densely labeled structures while enabling the quantification of uncertainty of SR results. Two types of uncertainty is considered, i.e. aleatoric uncertainty (AleaU) and epistemic uncertainty (EpisU).



## Requirements

The software was tested on a *Linux* system with Ubuntu version 20.04, and a *Windows* system with Windows 10 Home. 

### Hardware

+ Server
  + OS: Linux (Ubuntu 20.04)
  + CPU: Intel(R) Xeon(R) Silver 4210R CPU @ 2.40GHz
  + GPU: Nvidia GeForce GTX 2080 Ti, 11 GB memory
+ PC
  + OS: Windows 10
  + CPU: 11th Gen Intel(R) Core(TM) i5-11600KF @ 3.90GHz
  + GPU: Nvidia GeForce GTX 1080 Ti, 11 GB memory

### Software

Prerequisites:

+ Python 3.7
+ Matlab R2017a

The code is written in python and relies on pytorch. We use the python libraries as follows: 

+ Pytorch  1.11.0
+ Numpy 1.21.5
+ imageio 2.18.0
+ skimage 0.19.2
+ matplotlib 3.5.1
+ cv2 4.5.5
+ natsort 8.1.0
+ tifffile



## Installation

1. Install Anaconda3 following the instructions [online](https://www.anaconda.com/download).

2. Create environment:

   ```
   conda create -n BayesDL python=3.7
   ```

3. Activate the environment

   ```
   # for windows
   activate BayesDL
   
   # for Linux
   source activate BayesDL
   ```

4. Install the python dependencies via **conda** or **pip**. It is recommended to use **conda** to install ***pytorch***.

   ```
   # CUDA 10.2
   conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=10.2 -c pytorch
   # CUDA 11.3
   conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
   ```

   

## Quickstart (demo)

1. Clone this repository into any place you want.

   ```
   git clone https://github.com/BayesDL-SIM
   cd BayesDL-SIM
   ```

2. Run the script `demo.sh` to execute the demo. You can see the results in `./Data/F-actin/BayesDL-results` folder.

   ```
   sh demo.sh   # you are now in */BayesDL-SIM
   ```

   *The pretrained models for F-actin are in `./checkpoint/F-actin/pretrained_G` folder.

   *You can also modify the parameters in `demo.sh` to suit your needs.



## How to train BayesDL

+ We used [BioSR](https://www.nature.com/articles/s41592-020-01048-5) dataset to train our model. Please download it from [here](https://figshare.com/articles/dataset/BioSR/13264793) (11.62 GB).

+ Unpack the tar file to any place you want. Then, run the matlab script `./DatasetGenerate/DataGenerate.m` to generate the training dataset. You can change the `DataPath` and `SavePath` arguments in `./DatasetGenerate/DataGenerate.m` to suit your needs.

+ Change the ```root_dir``` argument in ```train.py``` to the place where the generated training dataset is located.

+ Start the training of BayesDL-SIM:

  ```
  # Step 1 of DeT for SIM reconstruction
  python train.py --model BayesDL --train_type '' --likehood '' --SGLD True
  
  # Step 2 of DeT for AleaU quantification
  python train.py --model BayesDL --train_type uncertainty_tail --likehood 'gauss' --SGLD False --epoch 300
  ```

+ The well-trained models are saved in the `./heckpoint/F-actin/BayesDL-SIM` folder. You can change the save path by modify the `--exp_name` argument in the `train.py`.



## Citation

Our paper is currently under review.



## License

This project is released under the Apache 2.0 license.



## Contact

To report any bugs, suggest improvements, or ask questions, please contact me at "hust_liutao@hust.edu.cn".