# Installation / Setup

Navigate to the  folder:

	cd efficientdet_comparison/install_evaluations

Run the installer:

	bash -i setup.sh

This will download the required repo-weights and adjust the dataset will be used for evaluation. It will also clone an EfficientDet implementation into the repo (albeit gitignored) for evaluation purposes.

# Running the detection model on a video 

Navigate to the root folder of nanotracker (same as this readme file). Activate the installed environment and run demo by giving mp4 video path:

    source ~/.bashrc
    source activate ~/venv_p39_nanotracker
    python video_demo.py --video_path {path_to_input_mp4_file}

Output will be saved in root folder with name "./out_{filename}.mp4"

# Running the EfficientDet comparison benchmark

## Evaluating EfficientDet on WIDER

Navigate to the EfficientDet evaluation folder:

    cd efficientdet_comparison/Yet-Another-EfficientDet-Pytorch

Activate the installed environment and run evaluation

    source ~/.bashrc
    source activate ~/venv_p39_nanotracker
    python coco_eval.py -p wider -c 0 -w ./weights/efficientdet-d0.pth

For evaluation on CPU, run the following instead:

    python coco_eval.py -p wider -c 0 -w ./weights/efficientdet-d0.pth --cuda False

## Evaluating Our Quantized NanoTracker model on WIDER

Navigate to the root folder of nanotracker (same as this readme file)

Activate the installed environment and run evaluation

    source ~/.bashrc
    source activate ./venv_p39_nanotracker
    python efficientdet_comparison/coco_eval.py -m qat -dp ./efficientdet_comparison/datasets/wider/val -ap ./efficientdet_comparison/datasets/wider/annotations/instances_val.json -wp ./efficientdet_comparison/training_experiment_best.pth.tar

