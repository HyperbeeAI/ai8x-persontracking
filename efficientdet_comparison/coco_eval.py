###########################################################################
# Computer vision - Embedded person tracking demo software by HyperbeeAI. #
# Copyrights © 2023 Hyperbee.AI Inc. All rights reserved. hello@hyperbee.ai #
###########################################################################
# Author: Zylo117

"""
COCO-Style Evaluations

put images here datasets/your_project_name/val_set_name/*.jpg
put annotations here datasets/your_project_name/annotations/instances_{val_set_name}.json
put weights here /path/to/your/weights/*.pth
change compound_coef

"""

import json
import os
import numpy as np

import argparse
import torch
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import time

from models import mnv2_SSDlite
from library.ssd import conv_model_fptunc2fpt, conv_model_fpt2qat, conv_model_qat2hw, collate_fn, PredsPostProcess, round_floats
from dataloader import CocoDetection, input_fxpt_normalize

#from library.ssd import generateAnchorsInOrigImage, collate_fn, point_form, prepareHeadDataforLoss_fast, plot_image_mnv2_2xSSDlite, sampleRandomPicsFromCOCO, saveOutputs ,PredsPostProcess, calculatemAP, batchNormAdaptation, round_floats

ap = argparse.ArgumentParser()
ap.add_argument('-m', '--mode', type=str, default='qat', help='Mode of the model, allowed modes: fpt_unc, fpt, qat')
ap.add_argument('--nms_threshold', type=float, default=0.5, help='non max supression threshold')
ap.add_argument('--conf_threshold', type=float, default=0.5, help='confidence treshold, predictions below this level will be discarded')
ap.add_argument('-dp', '--data_path', type=str, default=None, help='/path/to/images')
ap.add_argument('-ap', '--json_path', type=str, default=None, help='/path/to/annotations.json')
ap.add_argument('-wp', '--weights_path', type=str, default=None, help='/path/to/weights')

args = ap.parse_args()

mode = args.mode
nms_threshold = args.nms_threshold
conf_threshold = args.conf_threshold
data_path = args.data_path
json_path = args.json_path
weights_path = args.weights_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_coco(model, DATA_PATH, JSON_PATH , nmsIoUTreshold = 0.5, PredMinConfTreshold = 0.5, HW_mode = False):
    
    if HW_mode:
        act_8b_mode = True
    else:
        act_8b_mode = False

    transform = transforms.Compose([transforms.ToTensor(), input_fxpt_normalize(act_8b_mode=act_8b_mode)])
    targetFileName = 'resized.json'
    dataset = CocoDetection(root=DATA_PATH, annFile=JSON_PATH, transform=transform, scaleImgforCrop= None)

    dataset.createResizedAnnotJson(targetFileName=targetFileName)
    resizedFilePath = os.path.join(os.path.split(JSON_PATH)[0],targetFileName)
    cocoGt=COCO(resizedFilePath)
    os.remove(resizedFilePath)

    seq_sampler = torch.utils.data.SequentialSampler(dataset)
    data_loader = DataLoader(dataset,
                              sampler=seq_sampler,
                              batch_size=1,
                              collate_fn=collate_fn,
                              drop_last=False)
    print(f"Dataset Length: {len(dataset)}, Number of Batches: {len(data_loader)}")
    
    ANCHORS_HEAD1 = [(11.76, 28.97),
                     (20.98, 52.03),
                     (29.91, 77.24),
                     (38.97, 106.59)]

    ANCHORS_HEAD2 = [(52.25, 144.77),
                    (65.86, 193.05),
                    (96.37, 254.09),
                    (100.91, 109.82),
                    (140, 350)]
    
    predsPostProcess = PredsPostProcess(512, ANCHORS_HEAD1, ANCHORS_HEAD2)

    
    dataDictList =[]
    imgIDS = []
    for i, data in enumerate(tqdm(data_loader)):
        imageBatch, targetBatch , idxBatch = data

        imageStack = torch.stack(imageBatch).detach().to(device)
        imageStack.requires_grad_(True)
        predBatch = model(imageStack)
        
        if HW_mode:
            BBs1 = predBatch[0].detach() / 128.0
            CFs1 = predBatch[1].detach() / 128.0
            BBs2 = predBatch[2].detach() / 128.0
            CFs2 = predBatch[3].detach() / 128.0
        else:
            BBs1 = predBatch[0].detach()
            CFs1 = predBatch[1].detach()
            BBs2 = predBatch[2].detach()
            CFs2 = predBatch[3].detach()

        for imgNum in range(imageStack.shape[0]):
            img = imageStack[imgNum,:,:,:]
            target = targetBatch[imgNum]
            image_id = int(idxBatch[imgNum])
            imgIDS.append(image_id)

            pred = (BBs1[imgNum,:,:,:].unsqueeze(0), CFs1[imgNum,:,:,:].unsqueeze(0), 
                    BBs2[imgNum,:,:,:].unsqueeze(0), CFs2[imgNum,:,:,:].unsqueeze(0))

            boxes, confidences = predsPostProcess.getPredsInOriginal(pred)

            nms_picks   = torchvision.ops.nms(boxes, confidences, nmsIoUTreshold) 
            boxes_to_draw = boxes[nms_picks]
            confs_to_draw = confidences[nms_picks]
            confMask = (confs_to_draw > PredMinConfTreshold)

            # Inputs to mAP algorithm
            if (confMask.any()):

                # pred boxes -> [xmin,ymin,xmax,ymax], tensor shape[numpred,4]
                bbox = boxes_to_draw[confMask]
                scores = confs_to_draw[confMask]
                # Convert BB to coco annot format -> [xmin,ymin,width, height]
                bbox[:,2] = bbox[:,2] - bbox[:,0]
                bbox[:,3] = bbox[:,3] - bbox[:,1]
                

                bbox = bbox.tolist() # pred boxes -> [xmin,ymin,xmax,ymax], shape[numpred,4]
                score = scores.tolist()
                category_id = np.ones_like(score,dtype=int).tolist()

                for j in range(len(bbox)):
                    box = {"image_id":image_id, "category_id":category_id[j], "bbox":bbox[j],"score":score[j]}
                    dataDictList.append(round_floats(box))

    if (len(dataDictList)):        
        # Evavluate and Accumulate mAP for remained baches, if any
        cocoDT = json.dumps(dataDictList)

        # Write detections to .json file
        with open('cocoDT.json', 'w') as outfile:
            outfile.write(cocoDT) 

        # Load detections
        cocoDt=cocoGt.loadRes('cocoDT.json')
        os.remove("cocoDT.json")

        # running evaluation
        annType = 'bbox'
        cocoEval = COCOeval(cocoGt,cocoDt,annType)
        cocoEval.params.catIds = 1
        cocoEval.params.imgIds  = imgIDS
        cocoEval.evaluate()
        cocoEval.accumulate()
        
        print('')
        cocoEval.summarize()
    else:
        raise Exception('the model does not provide any valid output, check model architecture and the data input')


if __name__ == '__main__':
    model = mnv2_SSDlite()
    
    layer_bits_dictionary = {}
    layer_bits_dictionary['conv1' ] = 8;
    layer_bits_dictionary['epw_conv2' ] = 8;
    layer_bits_dictionary['dw_conv2' ]  = 8;
    layer_bits_dictionary['ppw_conv2' ] = 8;

    layer_bits_dictionary['epw_conv3' ] = 8;
    layer_bits_dictionary['dw_conv3' ]  = 8;
    layer_bits_dictionary['ppw_conv3' ] = 8;

    layer_bits_dictionary['epw_conv4' ] = 8;
    layer_bits_dictionary['dw_conv4' ]  = 8;
    layer_bits_dictionary['ppw_conv4' ] = 8;

    layer_bits_dictionary['epw_conv5']  = 8;
    layer_bits_dictionary['dw_conv5']   = 8;
    layer_bits_dictionary['ppw_conv5']  = 8;

    layer_bits_dictionary['epw_conv6']  = 8;
    layer_bits_dictionary['dw_conv6']   = 8;
    layer_bits_dictionary['ppw_conv6']  = 8;

    layer_bits_dictionary['epw_conv7']  = 8;
    layer_bits_dictionary['dw_conv7']   = 8;
    layer_bits_dictionary['ppw_conv7']  = 8;

    layer_bits_dictionary['epw_conv8']  = 8;
    layer_bits_dictionary['dw_conv8']   = 8;
    layer_bits_dictionary['ppw_conv8']  = 8;

    layer_bits_dictionary['epw_conv9']  = 8;
    layer_bits_dictionary['dw_conv9']   = 8;
    layer_bits_dictionary['ppw_conv9']  = 8;

    layer_bits_dictionary['epw_conv10'] = 8;
    layer_bits_dictionary['dw_conv10']  = 8;
    layer_bits_dictionary['ppw_conv10'] = 8;

    layer_bits_dictionary['epw_conv11'] = 8;
    layer_bits_dictionary['dw_conv11']  = 8;
    layer_bits_dictionary['ppw_conv11'] = 8;

    layer_bits_dictionary['epw_conv12'] = 8;
    layer_bits_dictionary['dw_conv12']  = 8;
    layer_bits_dictionary['ppw_conv12'] = 8;

    layer_bits_dictionary['epw_conv13'] = 8;
    layer_bits_dictionary['dw_conv13']  = 8;
    layer_bits_dictionary['ppw_conv13'] = 8;

    layer_bits_dictionary['epw_conv14'] = 8;
    layer_bits_dictionary['dw_conv14']  = 8;
    layer_bits_dictionary['ppw_conv14'] = 8;

    layer_bits_dictionary['epw_conv15'] = 8;
    layer_bits_dictionary['dw_conv15']  = 8;
    layer_bits_dictionary['ppw_conv15'] = 8;

    layer_bits_dictionary['epw_conv16'] = 8;
    layer_bits_dictionary['dw_conv16']  = 8;
    layer_bits_dictionary['ppw_conv16'] = 8;

    layer_bits_dictionary['epw_conv17'] = 8;
    layer_bits_dictionary['dw_conv17']  = 8;
    layer_bits_dictionary['ppw_conv17'] = 8;

    layer_bits_dictionary['epw_conv18'] = 8;
    layer_bits_dictionary['dw_conv18']  = 8;
    layer_bits_dictionary['ppw_conv18'] = 8;

    layer_bits_dictionary['head1_dw_classification'] = 8;
    layer_bits_dictionary['head1_pw_classification'] = 8;
    layer_bits_dictionary['head1_dw_regression'] = 8;
    layer_bits_dictionary['head1_pw_regression'] = 8;

    layer_bits_dictionary['head2_dw_classification'] = 8;
    layer_bits_dictionary['head2_pw_classification'] = 8;
    layer_bits_dictionary['head2_dw_regression'] = 8;
    layer_bits_dictionary['head2_pw_regression'] = 8;

    # Convert model to appropriate mode before loading weights
    HW_mode = False
    if mode == 'fpt_unc':
        model.to(device)
        
    elif mode == 'fpt':
        model = conv_model_fptunc2fpt(model)
        model.to(device)
        
    elif mode == 'qat':
        model = conv_model_fptunc2fpt(model)
        model.to(device)
        model = conv_model_fpt2qat(model, layer_bits_dictionary)
        model.to(device)
        
    elif mode == 'hw':
        HW_mode = True
        model = conv_model_fptunc2fpt(model)
        model.to(device)
        model = conv_model_fpt2qat(model, layer_bits_dictionary)
        model.to(device)
        model = conv_model_qat2hw(model)
        model.to(device)

    else:
        raise Exception('Invalid model mode is selected, select from: fpt_unc, fpt, qat, hw')


    weights = torch.load(weights_path, map_location=torch.device('cpu'))
    model.load_state_dict(weights['state_dict'], strict=True)

    model.requires_grad_(False)
    model.eval()
    
    if mode == 'qat' or mode == 'hw':
        print(''*5)
        print('*'*120)
        print('qat or hardware mode is selected, please make sure you configured layer_bits_dictionary in "coco_eval.py" accordingly!!!')
        print('*'*120)
        print('')
        time.sleep(5)

    evaluate_coco(model, DATA_PATH=data_path, JSON_PATH=json_path , nmsIoUTreshold=nms_threshold, 
                  PredMinConfTreshold=conf_threshold, HW_mode = HW_mode)