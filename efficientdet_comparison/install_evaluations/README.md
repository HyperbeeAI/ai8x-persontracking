# Evaluation Setup

1- Clone the repository

2- Navigate to the  folder:

	cd efficientdet_comparison/install_evaluations

3- Run the installer (This will download the required repo-weights and adjust the dataset will be used for evaluation):

	bash -i setup.sh


# Run evaluations

## Evaluating EfficientDet with Wider Person Validation Dataset

1- Navigate to the EfficientDet evaluation folder that is installed:

    cd efficientdet_comparison/Yet-Another-EfficientDet-Pytorch

2- Activate the installed environment and run evaluation
    source ~/.bashrc
    source activate ~/venv_p39_nanotracker
    python coco_eval.py -p wider -c 0 -w ./weights/efficientdet-d0.pth

Note: For evaluation on cuda disabled platform:
    python coco_eval.py -p wider -c 0 -w ./weights/efficientdet-d0.pth --cuda False

## Evaluating Our Quantized NanoTracker model with Wider Person Validation Dataset

1- Navigate to the root folder of nanotracker

2- Activate the installed environment and run evaluation
    source ~/.bashrc
    source activate ~/venv_p39_nanotracker
    python coco_eval.py -m qat -dp ./datasets/wider/val -ap ./datasets/wider/annotations/instances_val.json -wp ./efficientdet_comparison/training_experiment_best.pth.tar

