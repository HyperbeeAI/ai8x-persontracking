###########################################################################
# Computer vision - Embedded person tracking demo software by HyperbeeAI. #
# Copyrights © 2023 Hyperbee.AI Inc. All rights reserved. hello@hyperbee.ai #
###########################################################################
import os
import datetime, time
import json
from PIL import Image
from tqdm import tqdm

import torch, torchvision 
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset

from typing import Any, Callable, Optional, Tuple, List
import argparse

##############################################################################
####################### Functions to Prepare Images ##########################
##############################################################################
# Functions defined for prescaling images/targets before center crop operation
def calcPrescale(image_width, image_height, scaleImgforCrop = 512):
    # Calculate scale factor to shrink/expand image to coincide width or height to croppig area
    scale = 1.0
    # image fully encapsulates cropping area or vice versa
    if ((image_width-scaleImgforCrop)*(image_height-scaleImgforCrop) > 0): 
        # if width of original image is closer to crop area
        if abs(1-image_width/scaleImgforCrop) < abs(1-image_height/scaleImgforCrop):
            scale = image_width/scaleImgforCrop
        else:
            scale = image_height/scaleImgforCrop
    return scale


# Scales the image with defined scale
def prescaleImage(image, scale):
    
    image_width = int(image.size[0]/scale)
    image_height = int(image.size[1]/scale)


    image_res = image.resize((image_width, image_height))
    return image_res


def preProcessImages(org_images_path):
    corruptedImgs = []
    ccrop_size = 512
    folder_dir,folder_name = os.path.split(org_images_path)
    cur_dir = os.getcwd()
    
    processed_images_path = os.path.join(cur_dir,'datasets','wider','val')
    
    if not os.path.isdir(processed_images_path):
        os.makedirs(processed_images_path)
    imageNames = os.listdir(org_images_path)
    
    for i, image in enumerate(tqdm(imageNames)):
        try:
            if(image.split('.')[1] == 'jpg'):
                imgDir = os.path.join(org_images_path,image)
                img = Image.open(imgDir)

                # prescaling 
                image_width = img.size[0]
                image_height = img.size[1]
                scale = calcPrescale(image_width, image_height,scaleImgforCrop=ccrop_size)
                img_resized = prescaleImage(img, scale)

                # Center Crop
                width, height = img_resized.size   # Get dimensions

                left = (width - ccrop_size)/2
                top = (height - ccrop_size)/2
                right = (width + ccrop_size)/2
                bottom = (height + ccrop_size)/2

                # Crop the center of the image
                img_ccropped = img_resized.crop((left, top, right, bottom))
                img_ccropped.save(os.path.join(processed_images_path, image))
        except:
            print('Cannot Load: ' + image + ', check if it is corrupted.')
            corruptedImgs.append(image)

    print('')
    print('Conversion Finished')
    print('')
    if len(corruptedImgs):
        print('Something wrong with the following images and they are not processed:')
        print(corruptedImgs)
        print('Please delete these images from associated annotations')
    return


##############################################################################
##################### Functions to Prepare Annotations #######################
##############################################################################
class CocoDetection(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        
        annFile (string): Path to json annotation file.
        
        scaleImgforCrop (int, optional): Img and target BBs are scaled with 
        constant aspect ratio st:
            if image width, image height > scaleImgforCrop image is shrinked 
            until width or height becomes equal to scaleImgforCrop
    
            if image width, image height < scaleImgforCrop image is expanded 
            until width or height becomes equal to scaleImgforCrop
            
            else no scaling
        transform (callable, optional): A function/transform that  takes in an 
        PIL image and returns a transformed version. E.g, ``transforms.ToTensor``
        
        target_transform (callable, optional): A function/transform that takes in 
        the target and transforms it.
        transforms (callable, optional): A function/transform that takes input 
        sample and its target as entry and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        annFile: str,
        scaleImgforCrop: int= None, 
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None
    ):
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.annFilePath = os.path.join('.',annFile)
        self.catPersonId = self.coco.getCatIds(catNms=['person'])[0]
        self.scaleImgforCrop = scaleImgforCrop
        
    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id, iscrowd=False))

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:

        id = self.ids[index]
        imgID = id

        try:
            image = self._load_image(id)
        except:
            print(f'********Unable to load image with id: {imgID}********')
            print('Please check if image is corrupted, and remove it from annotations if necessary.')
            
            
        target = copy.deepcopy(self._load_target(id)) # deepcopy target list beforecentercrop manip, to be abe to work with same
                                                      # dateset without reloading it 
        
        image_width = image.size[0]
        image_height = image.size[1]
        
        
        # If necesary rescale the image and BBs near the size of planned center crop as much as possible
        scale = self._calcPrescale(image_width=image_width, image_height=image_height)
        image = self._prescaleImage(image, scale)
        
        for i, t in enumerate(target):
            BB = t['bbox'].copy()
            scaledBB = self._prescaleBB(BB,scale)
            target[i]['bbox'] = scaledBB
            
        
        
        # Image width height after prescaling
        image_width = image.size[0]
        image_height = image.size[1]
        
        # Check if center crop applied
        centerCropped = False
        if self.transforms is not None:
            image, target = self.transforms(image, target)
            
            # If center crop applied, transform BBs as well
            for t in self.transforms.transform.transforms:
                if (type(t) == torchvision.transforms.transforms.CenterCrop):
                    centerCropped = True
        
        

        x_scale = image.size(2) / image_width
        y_scale = image.size(1) / image_height

        bbox_arr = []
        
        for idx,ann in enumerate(target):
            if ann['category_id'] == self.catPersonId:
                crop_size = image.shape[1]
                
                if centerCropped:
                    bbox = ann['bbox'].copy()
                    croppedBB = self.cropBBox(bbox, crop_size,image_height,image_width)    
                else:
                    croppedBB = torch.tensor(ann['bbox']) 
                
                if not (croppedBB == None):
                    bbox_arr.append(croppedBB)                
                             
        if len(bbox_arr) != 0:
            bbox_arr = torch.stack(bbox_arr)
            wh = bbox_arr[:, 2:] 
            xy = bbox_arr[:, :2]
            
            id_tensor = torch.tensor([id]).unsqueeze(0).expand(bbox_arr.size(0), -1)

            bbox_arr = torch.cat([id_tensor, xy, wh], dim=-1)
        else:
            bbox_arr = torch.tensor(bbox_arr)
            
            
        return image, bbox_arr , imgID 

    def __len__(self) -> int:
        return len(self.ids)
    
    def get_labels(self):
        labels = []
        for id in self.ids:
            anns = self._load_target(id)
            person_flag = False
            for ann in anns:     
                person_flag = ann['category_id'] == self.catPersonId
                if person_flag == True:
                    break
            if person_flag == True:
                labels.append(1)
            else:
                labels.append(0)
        return torch.tensor(labels)
    
    def get_cat_person_id(self):
        return self.catPersonId
    
    def get_coco_api(self):
        return self.coco
    
    
    # Functions defined for prescaling images/targets before center crop operation
    def _calcPrescale(self, image_width, image_height):
        # Calculate scale factor to shrink/expand image to coincide width or height to croppig area
        scale = 1.0
        if self.scaleImgforCrop != None:
            # image fully encapsulates cropping area or vice versa
            if ((image_width-self.scaleImgforCrop)*(image_height-self.scaleImgforCrop) > 0): 
                # if width of original image is closer to crop area
                if abs(1-image_width/self.scaleImgforCrop) < abs(1-image_height/self.scaleImgforCrop):
                    scale = image_width/self.scaleImgforCrop
                else:
                    scale = image_height/self.scaleImgforCrop
        return scale
    
    # Scales the image with defined scale
    def _prescaleImage(self, image, scale):
        image_width = int(image.size[0]/scale)
        image_height = int(image.size[1]/scale)

        t = transforms.Resize([image_height,image_width])
        image = t(image)
        return image
    
    # Scales the targets with defined scale
    def _prescaleBB(self, BB, scale):
        scaledbb = [round(p/scale,1) for p in BB]
        return scaledbb
        
    
    def cropBBox(self,bbox,crop_size, image_height, image_width):

        bbox_aligned = []
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]

        # Casses for cropping       
        if image_height < crop_size:
            offset = (crop_size - image_height) // 2
            y = y + offset
            if (y+h) > crop_size:
                offset = (y+h)-crop_size
                h = h - offset
        if image_width < crop_size:
            offset = (crop_size - image_width) // 2
            x = x + offset
            if (x+w) > crop_size:
                offset = (x+w)-crop_size
                w = w - offset
        if image_width > crop_size:
            offset = (image_width - crop_size) // 2
            if offset > x:
                # Deal with BB coincide with left cropping boundary
                w = w -(offset-x)
                x = 0
            else:
                x = x - offset
                
                # Deal with BB coincide with right cropping boundary
                if (x+w) > crop_size:
                    offset = (x+w)-crop_size
                    w = w - offset
                    
        if image_height > crop_size:

            offset = (image_height - crop_size) // 2
            if offset > y:
                # Deal with BB coincide with top cropping boundary
                h = h -(offset-y)
                y = 0
            else:
                y = y - offset
                # Deal with BB coincide with bottom cropping boundary
                if (y+h) > crop_size:
                    offset = (y+h)-crop_size
                    h = h - offset
        
        bbox_aligned.append(x)
        bbox_aligned.append(y)
        bbox_aligned.append(w)
        bbox_aligned.append(h)

        if ((w <= 0) or (h <= 0)):
            return None
        else:
            x_scale, y_scale = 1.0,1.0
            return torch.mul(torch.tensor(bbox_aligned), torch.tensor([x_scale, y_scale, x_scale, y_scale]))
    
    def __round_floats(self,o):
        '''
        Used to round floats before writing to json file
        '''
        if isinstance(o, float):
            return round(o, 2)
        if isinstance(o, dict):
            return {k: self.__round_floats(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [self.__round_floats(x) for x in o]
        return o
    
    def createResizedAnnotJson(self,targetFileName,cropsize = 512):
        '''
        Resizes person annotations after center crop operation and saves as json file to the
        directory of original annotations with the name "targetFileName"
        '''
        t1 = time.time()
        # Get original json annot file path, and create pah for resized json annot file
        path, annotfilename = os.path.split(self.annFilePath)
        resizedAnnotPath = os.path.join(path,targetFileName)
        
        print('')
        print(f'Creating Json file for resized annotations: {resizedAnnotPath}')
        

        # Load original annotation json file as dictionary and assign it to resized annot dict
        with open(self.annFilePath) as json_file:
            resizedanotDict = json.load(json_file)

        # Original annotations array
        origannList = resizedanotDict['annotations']
        
        # Check if center crop applied
        centerCropped = False
        if self.transforms is not None:
            # If center crop applied, transform BBs as well
            for t in self.transforms.transform.transforms:
                if (type(t) == torchvision.transforms.transforms.CenterCrop):
                    centerCropped = True
        
        resizedannList = []
        for resizedannot in origannList:

            currentcatID = resizedannot['category_id']
            currentBB = resizedannot['bbox']
            currentImgID = resizedannot['image_id']
            
            
            # Get crop size and original image sizes
            image_width = self.coco.loadImgs(currentImgID)[0]['width']
            image_height = self.coco.loadImgs(currentImgID)[0]['height']
            
            # If presclae applied to image, calculate new image width and height
            scale = self._calcPrescale(image_width=image_width, image_height=image_height)
            image_width = image_width / scale
            image_height = image_height / scale

            if currentcatID == self.catPersonId:
                # if BB is person
                bbox = resizedannot['bbox'].copy()
                
                # If prescale appied to image, resize annotations BBs
                bbox = self._prescaleBB(bbox, scale)
                
                # If center crop  applied, crop/recalculate BBs as well
                if centerCropped:
                    croppedBB = self.cropBBox(bbox, cropsize,image_height,image_width)    
                else:
                    croppedBB = torch.tensor(bbox) 
                
                if (croppedBB != None):
                    # If BB is person and valid after crop, add it to resized annotations list
                    croppedBB = croppedBB.tolist()
                    resizedannot['bbox'] = self.__round_floats(croppedBB)
                    resizedannot['area'] = self.__round_floats(croppedBB[2]*croppedBB[3])
                    resizedannList.append(resizedannot)
            else:
                # If BB is non-person add it to resized annotations list as it is
                resizedannList.append(resizedannot)   

        resizedanotDict['annotations'] = resizedannList
        print('Saving resized annotations to json file...')

        # Save resized annotations in json file
        resizedanotDict = json.dumps(resizedanotDict)
        with open(resizedAnnotPath, 'w') as outfile:
            outfile.write(resizedanotDict)

        print(f'{resizedAnnotPath} saved.')
        t2 = time.time()
        print(f'Elapsed time: {t2-t1} seconds')
        
# Taken from : https://github.com/hasanirtiza/Pedestron/blob/master/tools/convert_datasets/pycococreatortools.py
def create_image_info(image_id, file_name, image_size,
                      date_captured=datetime.datetime.utcnow().isoformat(' '),
                      license_id=1, coco_url="", flickr_url=""):

    image_info = {
            "id": image_id,
            "file_name": file_name,
            "width": image_size[0],
            "height": image_size[1],
            "date_captured": date_captured,
            "license": license_id,
            "coco_url": coco_url,
            "flickr_url": flickr_url
    }

    return image_info

# Taken from : https://github.com/hasanirtiza/Pedestron/blob/master/tools/convert_datasets/pycococreatortools.py
def create_annotation_info(annotation_id, image_id, category_info, bounding_box):
    is_crowd = category_info['is_crowd']

    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_info["id"],
        "iscrowd": is_crowd,
        "bbox": bounding_box
    }

    return annotation_info

def convWidertoCOCO(annotFile, orgImageDir):
    '''
    Converts wider dataset annotations to COCO format. 
    Args: 
        annotFile: Original annotation file
        orgImageDir: Original Images directory
    '''

    totalImgnum = 0
    imgID = 0
    annID = 0

    imgList = []
    annList = []

    category_info= {}
    category_info['is_crowd'] = False
    category_info['id'] = 1

    data ={}

    data['info'] = {'description': 'Example Dataset', 'url': '', 'version': '0.1.0', 'year': 2022, 'contributor': 'ljp', 'date_created': '2019-07-18 06:56:33.567522'}
    data['categories'] = [{'id': 1, 'name': 'person', 'supercategory': 'person'}]
    data['licences'] = [{'id': 1, 'name': 'Attribution-NonCommercial-ShareAlike License', 'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/'}]

    with open(annotFile) as f:
        for _, annot_raw in enumerate(tqdm(f)):
            imgID += 1

            annot_raw = annot_raw.split()
            imgName = annot_raw[:1][0]

            totalImgnum +=1 
            imageFullPath = os.path.join(orgImageDir,imgName)
            try:
                curImg = Image.open(imageFullPath)
                image_size = curImg.size

                BBs_str = annot_raw[1:]
                bb_raw = [int(bb) for bb in BBs_str]

                imgInf = create_image_info(image_id = imgID, file_name = imgName, image_size =image_size,
                      date_captured=datetime.datetime.utcnow().isoformat(' '),
                      license_id=1, coco_url="", flickr_url="")
                imgList.append(imgInf)

                bb = []
                for i, p in enumerate(bb_raw):

                    bb.append(p)
                    if ((i+1)%4 == 0):
                        annID += 1
                        ann = create_annotation_info(annID, imgID, category_info = category_info, bounding_box = bb)
                        annList.append(ann)
                        bb = []

            except:
                print(f'Cannot create annot for {imgName}, image does not exist in given directory.')

    data['annotations'] = annList
    data['images'] = imgList
    
    
    cur_dir = os.getcwd()
    processed_annot_path = os.path.join(cur_dir,'datasets','wider','annotations')
    
    if not os.path.isdir(processed_annot_path):
        os.makedirs(processed_annot_path)
    
    orgCOCOAnnotFile = os.path.join( processed_annot_path ,'orig_annot.json')

    with open(orgCOCOAnnotFile, 'w') as fp:
        json.dump(data, fp)
    
    
    print('Annotations saved as: ' + orgCOCOAnnotFile)
    print(f'Created {annID} COCO annotations for total {totalImgnum} images')
    print('')
    return orgCOCOAnnotFile


def main():
    parser = argparse.ArgumentParser(description='This script converts original Wider Person'
                                                     'Validation Dataset images to 512 x 512'
                                                     'Then resisez the annotations accordingly, saves new images and annotations under datasets folder')
    parser.add_argument('-ip', '--wider_images_path', type=str, required = True,
                        help='path of the folder containing original images')
    parser.add_argument('-af', '--wider_annotfile', type=str, required = True,
                        help='full path of original annotations file e.g. ./some/path/some_annot.json')


    args = parser.parse_args()
    wider_images_path = args.wider_images_path
    wider_annotfile = args.wider_annotfile

    # Prepare images
    print('')
    print('Prescaling and Center-cropping original images to 512 x 512')
    preProcessImages(wider_images_path)
    print('\n'*2)

    # Convert original wider annotations in to COCO format
    print('Converting original annotations to COCO format')
    orgCOCOAnnotFile = convWidertoCOCO(wider_annotfile, wider_images_path)
    print('\n'*2)

    # Prescale/Center-crop annotations and save
    print('Prescaling/Center-cropping original annotations in COCO format')
    transform = transforms.Compose([transforms.CenterCrop(512), transforms.ToTensor()])
    dataset = CocoDetection(root=wider_images_path, annFile=orgCOCOAnnotFile, transform=transform,scaleImgforCrop= 512)
    targetFileName = 'instances_val.json'
    dataset.createResizedAnnotJson(targetFileName=targetFileName)
    os.remove(orgCOCOAnnotFile)

if __name__ == '__main__':
    main()
