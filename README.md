# torch--SP-RAN-Self-paced-Residual-Aggregated-Network
# SP-RAN
Source code for "**SP-RAN: Self-Paced Residual Aggregated Network for Solar Panel Mapping in Weakly Labeled Aerial Images
**", accepted in TGRS. The paper's PDF can be found in [Here](https://ieeexplore.ieee.org/document/9585690).

Jue Zhang; Xiuping Jia; Jiankun Hu

School of Engineering and Information Technology, University of New South Wales, Canberra

![image](https://github.com/zhangjue1993/torch--SP-RAN-Self-paced-Residual-Aggregated-Network/blob/main/Flowchart.png)

## Prerequisites
### environment
  - Windows 10
  - Torch 1.12.0
  - CUDA 11.6.0
  - Python 3.7.13
  - Opencv 3.4.2

### data sets
GoogleEarth Static Map API

# Parameters
Please refer to ```act_config.json```

## Training
### 1st training stage
GradCAM

### 2st training stage
 Setting the training data to the proper root as follows:
```
-- data -- train -- fore

                 -- back
                 
                 -- Pseudo labels
                
         -- test -- cls -- fore
         
                        -- back
                        
                 -- seg -- img
                 
                        -- gt
```
set ```self-pace``` to ```False``` in ```act_config.json```. 

Run ```python train.py --config_path act_config.json```.

### 3nd training stage

Set ```self-pace``` to ```True``` in ```act_config.json```. Set the label update dir in ```act_config.json```. Run  ```python train.py --config_path act_config.json```.

## Testing
```python predict.py --config_path act_config.json```

**Noting** that the results in our paper do not adopt any post-process including CRF.

The evaluation code can be found in [here](https://github.com/jiwei0921/Saliency-Evaluation-Toolbox).


## Contact me
If you have any questions, pleas feel free to contact me: jue.zhang@adfa.edu.au.


## Citation
We really hope this repo can contribute the conmunity, and if you find this work useful, please use the following citation:
```
@ARTICLE{9585690,
  author={Zhang, Jue and Jia, Xiuping and Hu, Jiankun},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={SP-RAN: Self-Paced Residual Aggregated Network for Solar Panel Mapping in Weakly Labeled Aerial Images}, 
  year={2022},
  volume={60},
  number={},
  pages={1-15},
  doi={10.1109/TGRS.2021.3123268}}
