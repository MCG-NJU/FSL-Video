## A Closer Look at Few-Shot Video Classification: A New Baseline and Benchmark
This repo contains the reference source code for the paper:

https://arxiv.org/abs/2110.12358

## Data Preparation
Few-shot versions of Kinetics and Something-Something V2 datasets can be downloaded from [here](https://box.nju.edu.cn/d/dc97163752fc4024be2c/). We used the split from [CMN](https://github.com/ffmpbgrnn/CMN/tree/master/kinetics-100) for Kinetics and the split from [OTAM](https://drive.google.com/drive/u/1/folders/1eyQmM2ZPXYOH_tuvseFP7yHg7tnuixqw) for SSv2. If you already have the full versions of Kinetics and SSv2, you can also use ```./tools/select_kinetics100.py``` to select the few-shot verison datasets. Generate the annotation using ```./tools/write_kinetics100.py```

## Feature Extractor Training
For classifier-based methods, we use the standard ResNet50 backbone and training strategies for video classification. Please refer to https://github.com/liu-zhy/temporal-adaptive-module for the feature extractor training. Note that we apply dropout for Baseline Plus and set 'consensus_type=avg' for both classifier-based methods. 

For meta-learning methods, modify the corresponding code to have the correct path and filename for the dataset. To train the Meta-Baseline for example (see paper for other hyperparams), run:

    CUDA_VISIBLE_DEVICES='0' python proto.py --work_dir [WORK_DIR] --dataset somethingotam

## Testing
For classifier-based methods, modify ```./config/test_baseline.yaml``` and run:

    CUDA_VISIBLE_DEVICES='0' python baseline_evaluate.py

For meta-learning methods, run:

    CUDA_VISIBLE_DEVICES='0' python proto.py --test_model True --checkpoint [CHECKPOINT] --dataset somethingotam

Please refer to utils.py for additional options.

## References
We have modified and integrated the following code into this project:

https://github.com/wyharveychen/CloserLookFewShot 

https://github.com/liu-zhy/temporal-adaptive-module 

https://github.com/wangzehui20/OTAM-Video-via-Temporal-Alignment 
