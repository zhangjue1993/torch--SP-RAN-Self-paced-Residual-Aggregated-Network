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

```
-- data -- train -- fore

                 -- back
                 
                 -- PLs
                
         -- test -- cls -- fore
                        --back
                 -- seg -- img
                        -- gt
     
```

## Training
### 1st training stage

Case2: We also upload ready-made pseudo labels in **Training data** (the link above), you can directly use our offered two kinds of pseudo labels for convenience. CAMs are also presented if you needed.

### 2nd training stage

#### 1, setting the training data to the proper root as follows:

```
MF_code -- data -- DUTS-Train -- image -- 10553 samples

                -- ECSSD (not necessary) 
                
                -- pseudo labels -- label0_0 -- 10553 pseudo labels
                
                                 -- label1_0 -- 10553 pseudo labels
```
#### 2, training
```Run main.py```

Here you can set ECCSD dataset as validation set for optimal results by setting ```--val``` to ```True```, of course it is not necessary in our work.

## Testing
```Run test_code.py```

You need to configure your desired testset in ```--test_root```.  Here you can also perform PAMR and CRF on saliency maps for a furthur refinements if you want, by setting ```--pamr``` and ```--crf``` to True. **Noting** that the results in our paper do not adopt these post-process for a fair comparison.

The evaluation code can be found in [here](https://github.com/jiwei0921/Saliency-Evaluation-Toolbox).


## Contact me
If you have any questions, pleas fell free to contact me: jue.zhang@adfa.edu.au.


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
