import sys
sys.path.append('pytorch-yolo-v3')
from preprocess import letterbox_image
import torch 
import torch.nn as nn
from torch.autograd import Variable
import cv2 
from util import *
import os 
import os.path as osp
from darknet import Darknet
import random 
import itertools
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class AnimeHeadDetector:
    def __init__(self, cfgfile, weightsfile):
        self.CONFIDENCE_THRESHOLD = 0.85
        self.NMS_THRESHOLD = 0.4
        self.NUM_CLASSES = 2 # hard code here for head detector
        self.CLASSES = [ 'Head' ]
        self.CUDA = torch.cuda.is_available()
        if self.CUDA:
            logging.info('Using CUDA.')
        else:
            logging.info('Using CPU.')
        logging.info("Loading network.....")
        self.model = Darknet(cfgfile)
        self.model.load_weights(weightsfile)
        self.model.net_info["height"] = 512 # hard code here because we didn't use Spp
        self.inp_dim = int(self.model.net_info["height"])
        if self.CUDA:
            self.model.cuda()
        self.model.eval()
        logging.info("Network successfully loaded.")

    # Detect the heads in the given image (opencv numpy array), and return the results
    def detect(self, image):
        # Preprocess the image
        w, h = image.shape[1], image.shape[0]
        img = (letterbox_image(image, (self.inp_dim, self.inp_dim)))
        img_ = img[:,:,::-1].transpose((2,0,1)).copy()
        img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
        im_dim_list = torch.FloatTensor([ [w, h] ]).repeat(1,2)

        # Send to the model for prediction
        if self.CUDA:
            img_ = img_.cuda()
        with torch.no_grad():
            prediction = self.model(Variable(img_), self.CUDA)
        output = write_results(prediction, self.CONFIDENCE_THRESHOLD, self.NUM_CLASSES, nms=True, nms_conf=self.NMS_THRESHOLD) # This function does NMS and converts the format. Returns 0 if no results are found.
        # output has the format of [ class, cx, cy, w, h, confidence ]

        if type(output) == int:
            return None
        
        # Convert back to the coordinates in the original image before resizing
        # We need somewhat complicated processing here because letter boxing was used in preprocessing
        output = output.detach().cpu()
        scaling_factor = torch.min(self.inp_dim/im_dim_list,1)[0].view(-1,1)
        
        output[:,[1,3]] -= (self.inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
        output[:,[2,4]] -= (self.inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2
        output[:,1:5] /= scaling_factor
        
        for i in range(output.shape[0]):
            output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[0,0])
            output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[0,1])

        return [ { 'l': int(output[i, 1].item()), 
            't': int(output[i, 2].item()),
            'r': int(output[i, 3].item()),
            'b': int(output[i, 4].item()), 
            'confidence': output[i, 5].item() } for i in range(output.shape[0]) ]

    # Detect heads and return the cropped faces as a list of opencv numpy arrays
    def detectAndCrop(self, image):
        results = self.detect(image)
        return [ image[result['t']: result['b'] + 1, result['l']: result['r'] + 1, :] for result in results ]

    # Detect heads and return the visualized face as a opencv numpy array
    def detectAndVisualize(self, image):
        results = self.detect(image)
        for result in results:
            cv2.rectangle(image, (result['l'], result['t']), (result['r'], result['b']), (0, 255, 0), 5)
        return image
