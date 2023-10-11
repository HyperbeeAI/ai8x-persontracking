import argparse
import os
import cv2
import sys
import numpy as np

import argparse
import torch, torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from models import mnv2_SSDlite 
from library.ssd import conv_model_fptunc2fpt, conv_model_fpt2qat, PredsPostProcess

from library.trackers.sort_tracker import Sort
from library.trackers.utils import draw_BBs, project_BBs_to_original_frame

root_path = os.path.dirname(os.path.realpath(__file__))

WEIGHTS_PATH = os.path.join(root_path, "./efficientdet_comparison/training_experiment_best.pth.tar")

def load_model(weights_path, mode='qat'):
    '''
    Loads the model in specified format, do not forget to 
    update weights and layer structure according to your 
    mnv2_SSDlite model and weight library
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mnv2_SSDlite()
    
    weight_dictionary = {}
    weight_dictionary['conv1' ] = 8
    weight_dictionary['epw_conv2' ] = 8
    weight_dictionary['dw_conv2' ]  = 8
    weight_dictionary['ppw_conv2' ] = 8

    weight_dictionary['epw_conv3' ] = 8
    weight_dictionary['dw_conv3' ]  = 8
    weight_dictionary['ppw_conv3' ] = 8

    weight_dictionary['epw_conv4' ] = 8
    weight_dictionary['dw_conv4' ]  = 8
    weight_dictionary['ppw_conv4' ] = 8

    weight_dictionary['epw_conv5']  = 8
    weight_dictionary['dw_conv5']   = 8
    weight_dictionary['ppw_conv5']  = 8

    weight_dictionary['epw_conv6']  = 8
    weight_dictionary['dw_conv6']   = 8
    weight_dictionary['ppw_conv6']  = 8

    weight_dictionary['epw_conv7']  = 8
    weight_dictionary['dw_conv7']   = 8
    weight_dictionary['ppw_conv7']  = 8

    weight_dictionary['epw_conv8']  = 8
    weight_dictionary['dw_conv8']   = 8
    weight_dictionary['ppw_conv8']  = 8

    weight_dictionary['epw_conv9']  = 8
    weight_dictionary['dw_conv9']   = 8
    weight_dictionary['ppw_conv9']  = 8

    weight_dictionary['epw_conv10'] = 8
    weight_dictionary['dw_conv10']  = 8
    weight_dictionary['ppw_conv10'] = 8

    weight_dictionary['epw_conv11'] = 8
    weight_dictionary['dw_conv11']  = 8
    weight_dictionary['ppw_conv11'] = 8

    weight_dictionary['epw_conv12'] = 8
    weight_dictionary['dw_conv12']  = 8
    weight_dictionary['ppw_conv12'] = 8

    weight_dictionary['epw_conv13'] = 8
    weight_dictionary['dw_conv13']  = 8
    weight_dictionary['ppw_conv13'] = 8

    weight_dictionary['epw_conv14'] = 8
    weight_dictionary['dw_conv14']  = 8
    weight_dictionary['ppw_conv14'] = 8

    weight_dictionary['epw_conv15'] = 8
    weight_dictionary['dw_conv15']  = 8
    weight_dictionary['ppw_conv15'] = 8

    weight_dictionary['epw_conv16'] = 8
    weight_dictionary['dw_conv16']  = 8
    weight_dictionary['ppw_conv16'] = 8

    weight_dictionary['epw_conv17'] = 8
    weight_dictionary['dw_conv17']  = 8
    weight_dictionary['ppw_conv17'] = 8

    weight_dictionary['epw_conv18'] = 8
    weight_dictionary['dw_conv18']  = 8
    weight_dictionary['ppw_conv18'] = 8

    weight_dictionary['head1_dw_classification'] = 8
    weight_dictionary['head1_pw_classification'] = 8
    weight_dictionary['head1_dw_regression'] = 8
    weight_dictionary['head1_pw_regression'] = 8

    weight_dictionary['head2_dw_classification'] = 8
    weight_dictionary['head2_pw_classification'] = 8
    weight_dictionary['head2_dw_regression'] = 8
    weight_dictionary['head2_pw_regression'] = 8


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
        model = conv_model_fpt2qat(model, weight_dictionary)
        model.to(device)
        
    else:
        raise Exception('Invalid model mode is selected, select from: fpt_unc, fpt, qat')

    weights = torch.load(weights_path, map_location=torch.device('cpu'))
    try:
        model.load_state_dict(weights['state_dict'], strict=True)
    except:
        print('Weights can not be loaded to model, please check compatibility of your layer definitions against weight dictionary. Check load_model() function.')

    model.requires_grad_(False)
    model.eval()
    return model

# k-Means Clustered Anchors
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

# Before initialization 
main_model = None   
main_model_weights_path = None  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

def initialize_model(weights_path, mode="qat"):
    global main_model_weights_path  
    if not main_model_weights_path or main_model_weights_path!=weights_path:    
        global main_model   
        main_model = load_model(weights_path, mode) 
        main_model_weights_path = weights_path    

def process_video(
        video_path,
        output_path,
        weights_path,
        mode="qat",
        nms_threshold=0.5,
        conf_threshold=0.5,
        max_age=10,
        min_hits=3,
        record_detections=False,
        show_track_confidences=False,
        enhanced_detector=False,
    ):

    initialize_model(weights_path, mode)    

    model = main_model
      
    network_input_size = (512,512) # width, height

    mot_tracker = Sort(max_age=max_age, 
                   min_hits=min_hits,
                   iou_threshold=nms_threshold)

    cap = cv2.VideoCapture(video_path)

    # get original video fps
    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    # With webcam get(CV_CAP_PROP_FPS) does not work.
    # Let's see for ourselves.
    if int(major_ver)  < 3 :
        output_fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    else :
        output_fps = cap.get(cv2.CAP_PROP_FPS)

    total_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    # Check if camera opened successfully
    if (cap.isOpened()== False):
        print("Error opening video file, please check video path and format. Exiting.")
        sys.exit()
        
    # Read until video is completed
    print('Generating detection video...')
    numframe = 0
    pbar = tqdm(total=total_length)
    while(cap.isOpened()):
        numframe += 1
        ret, frame = cap.read()

        if ret == True:

            # Get original resolution and calculate scale factor
            if numframe == 1:
                w = frame.shape[1]
                h = frame.shape[0]

                # Initiate writer w.r.t input video size
                if record_detections:
                    writer = cv2.VideoWriter(output_path, fourcc, output_fps, (w*2, h))
                else:
                    writer = cv2.VideoWriter(output_path, fourcc, output_fps, (w, h))

                scale = max(w/network_input_size[0], h/network_input_size[1])
                newsize = (int(w/scale), int(h/scale))


            resized_frame = cv2.resize(frame, newsize, fx=0, fy=0, interpolation=cv2.INTER_CUBIC)

            delta_w = network_input_size[1] - newsize[0]
            delta_h = network_input_size[0] - newsize[1]
            top, bottom = delta_h//2, delta_h-(delta_h//2)
            left, right = delta_w//2, delta_w-(delta_w//2)
            color = [0, 0, 0]
            padded_frame = cv2.copyMakeBorder(resized_frame, top, bottom, left, right, cv2.BORDER_CONSTANT,
                value=color)


            # Run model
            img = transforms.ToTensor()(padded_frame).unsqueeze(0).to(device)
            preds = model(img)
            BBs1 = preds[0].detach()
            CFs1 = preds[1].detach()
            BBs2 = preds[2].detach()
            CFs2 = preds[3].detach()


            pred = (BBs1[0,:,:,:].unsqueeze(0), CFs1[0,:,:,:].unsqueeze(0), 
                    BBs2[0,:,:,:].unsqueeze(0), CFs2[0,:,:,:].unsqueeze(0))

            absolute_boxes, person_cls = predsPostProcess.getPredsInOriginal(pred)
            nms_picks   = torchvision.ops.nms(absolute_boxes, person_cls, nms_threshold)

            # Filter by nms
            det_boxes = absolute_boxes[nms_picks]
            det_confs = person_cls[nms_picks]

            # Filter by conf
            cnf_picks = det_confs > conf_threshold
            det_boxes = det_boxes[cnf_picks]
            det_confs = det_confs[cnf_picks]

            det_list = torch.cat((det_boxes, det_confs.view(-1,1)),1).cpu().numpy()

            # Get Tracks
            trackers, track_confs = mot_tracker.update(det_list)
            track_confs = torch.tensor(track_confs, dtype=float)
            IDs = [int(x) for x in trackers[:,-1]]
            track_boxes = torch.tensor(trackers[:,0:4])


            # Draw BBs on the frame
            if record_detections:
                det_boxes_proj = project_BBs_to_original_frame(det_boxes, delta_w//2, delta_h//2, scale)
                frameDet= draw_BBs(det_boxes_proj, det_confs, np.copy(frame))

            track_boxes_proj = project_BBs_to_original_frame(track_boxes, delta_w//2, delta_h//2, scale)
            
            if enhanced_detector:
                frameTrack= draw_BBs(track_boxes_proj, track_confs, np.copy(frame))
            else:
                frameTrack= draw_BBs(track_boxes_proj, track_confs, np.copy(frame), 
                                     IDs= IDs, show_track_confidences=show_track_confidences)
                

            # Construct side to side frame (detections, tracks)
            if record_detections:
                frame = np.concatenate((frameDet,frameTrack),1)
            else:
                frame = frameTrack

            # Display the resulting frame
            writer.write(frame)

            # Press Q on keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            if numframe == -1:
                break
            pbar.update(1)


        # Break the loop
        else:
            break

    # When everything done, release
    # the video capture object
    cap.release()
    writer.release()

    # Closes all the frames
    cv2.destroyAllWindows()
    print('Done!')


def main():
    parser = argparse.ArgumentParser(description="A script to process video files and found bboxes.")
    parser.add_argument('--video_path', required=True, type=str, help="Path to the video file")
    args = parser.parse_args()

    video_path = args.video_path

    # Check if the video file exists
    if not os.path.exists(video_path):
        print(f" '{video_path}' does not exist.")
        exit(1) 

    print(f"Validated video path: {video_path}")

    # Use os.path.basename to get the filename with extension
    filename_with_extension = os.path.basename(video_path)
    filename, file_extension = os.path.splitext(filename_with_extension)
    if file_extension != ".mp4":
        print(f"File extension must be mp4")
        exit(1) 
    output_path = f"./out_{filename}.mp4"
    process_video(
        video_path=video_path, 
        output_path=output_path,
        weights_path=WEIGHTS_PATH, 
        max_age=4,
        min_hits=2,
        conf_threshold=0.5,
        enhanced_detector=True
    )

# Check if the script is being run directly (not imported as a module)
if __name__ == "__main__":
    main()