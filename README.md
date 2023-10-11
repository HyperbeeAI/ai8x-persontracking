# NanoTracker - person tracking with quantized networks by HyperbeeAI

Copyrights © 2023 Hyperbee.AI Inc. All rights reserved. hello@hyperbee.ai

This repository contains the quantized neural network-based person tracking utility by HyperbeeAI, NanoTracker. The algorithm is benchmarked against the WIDER pedestrian surveillance dataset and compared to the SOTA compact detection model EfficientDet.

A demo video of the detection model + a SORT tracker algorithm is shown below:

![demo](./experiments/demo.gif)

## Installation / Setup

Run the installer:

	bash -i setup.sh

This will download the required repo-weights and adjust the dataset will be used for evaluation. It will also clone an EfficientDet implementation into the repo (albeit gitignored) for evaluation purposes. Restart your shell after running this for changes to take effect (and for you to start being able to use the env).

## Running the detection model on a video 

Navigate to the root folder of nanotracker (same as this readme file). Activate the installed environment and run demo by giving mp4 video path:

    source ~/.bashrc
    conda activate ./venv_p39_nanotracker
    python video_demo.py --video_path {path_to_input_mp4_file}

Output will be saved in root folder with name "./out_{filename}.mp4"

## Running the EfficientDet comparison benchmark

### Evaluating EfficientDet on WIDER

Navigate to the EfficientDet evaluation folder:

    cd efficientdet_comparison/Yet-Another-EfficientDet-Pytorch

Activate the installed environment and run evaluation

    source ~/.bashrc
    conda activate ../../venv_p39_nanotracker
    python coco_eval.py -p wider -c 0 -w ./weights/efficientdet-d0.pth

For evaluation on CPU, run the following instead:

    python coco_eval.py -p wider -c 0 -w ./weights/efficientdet-d0.pth --cuda False

### Evaluating Our Quantized NanoTracker model on WIDER

Navigate to the efficientdet_comparison/ folder

    cd efficientdet_comparison/

Activate the installed environment and run evaluation

    source ~/.bashrc
    conda activate ./venv_p39_nanotracker
    python coco_eval.py -m qat -dp datasets/wider/val -ap datasets/wider/annotations/instances_val.json -wp training_experiment_best.pth.tar
