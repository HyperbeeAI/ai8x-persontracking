# Detailed Info on the EfficientDet vs. NanoTracker Benchmark on WIDER

This directory is built for comparison of our "quantization aware trained" detection algorithm with one of the SOTA compact detection algorithms, EfficientDet-d0, which has comparable complexity and structure with our quantized model. Our person tracking algorithm uses MobileNet-v2 as backbone mechanism and combines it with 2 SSD heads using total of 9 anchor boxes. Overall model consists of 60 convolution layers. 

We have re-scaled and center cropped the images in [Wider Person Dataset](https://competitions.codalab.org/competitions/20132#learn_the_details), also we resized its annotations and converted in to COCO annotation format to use them in evaluation. 

## Dependencies (see ../setup.sh for more)
* [PyTorch](https://github.com/pytorch/pytorch)
* [Torchvision](https://github.com/pytorch/vision)
* [Pycocotools](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools)
* [webcolors](https://pypi.org/project/webcolors/)
* [PyYAML](https://github.com/yaml/pyyaml)

### About the WIDER Dataset
The dataset requires sign-up before you can use it. We do not re-distribute this dataset, and we only use it for evaluation here.

* Sign up [Codalab](https://competitions.codalab.org/) and  participate to [WIDER Face & Person Challenge 2019](https://competitions.codalab.org/competitions/20132)
* Under "Participate" tab click "Train & Validation Data in Google Drive" and download
    * val_data.tar.gz
    * Annotations/val_bbox.txt
* Extract val_data.tar.gz as val_data and move val_data folder under ./data/original_wider/val_data
* Move "val_bbox.txt" under ./data/original_wider/

## mAP Comparisons
### EfficientDet-d0
    ### Wider Validation Dataset mAP scores ###
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.292
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.543
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.275
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.109
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.409
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.532
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.106
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.369
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.435
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.270
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.546
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.678
    

### NanoTracker
    ### Wider Validation Dataset mAP scores ###
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.281
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.457
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.310
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.075
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.406
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.582
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.107
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.324
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.331
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.110
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.481
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.637
    