import sys
sys.path.append('pytorch-yolo-v3')
import argparse
import cv2
import torch 
import os 
import os.path as osp
from tqdm import tqdm
from AnimeHeadDetector import AnimeHeadDetector

def arg_parse():
    """
    Parse arguements to the detect module
    """
    
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
   
    parser.add_argument("--images", dest = 'images', help = 
                        "Image / Directory containing images to perform detection upon",
                        default = "imgs", type = str)
    parser.add_argument("--det", dest = 'det', help = 
                        "Image / Directory to store detections to",
                        default = "det", type = str)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "head.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "head.weights", type = str)
    
    return parser.parse_args()

if __name__ ==  '__main__':
    args = arg_parse()
    detector = AnimeHeadDetector(args.cfgfile, args.weightsfile)
    images = args.images
    
    try:
        imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images) if os.path.splitext(img)[1] == '.png' or os.path.splitext(img)[1] =='.jpeg' or os.path.splitext(img)[1] =='.jpg']
    except NotADirectoryError:
        imlist = []
        imlist.append(osp.join(osp.realpath('.'), images))
    except FileNotFoundError:
        print ("No file or directory with the name {}".format(images))
        exit()
    if not os.path.exists(args.det):
        os.makedirs(args.det)
        
    for imfn in tqdm(imlist):
        im = cv2.imread(imfn)
        im2 = detector.detectAndVisualize(im) # pillow image
        cv2.imwrite(osp.join(args.det, osp.basename(imfn)), im2)

        # Sample usage of detectAndCrop
        #imgs = detector.detectAndCrop(im)
        #for i, img in enumerate(imgs):
            #cv2.imwrite(osp.join(args.det, '{}_{}.jpg'.format(osp.basename(imfn), i)), img)
    
    torch.cuda.empty_cache()
