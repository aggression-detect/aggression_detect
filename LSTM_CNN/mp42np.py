import cv2
import numpy as np
#import matplotlib.pyplot as plt
import glob 
from sys import argv
import os
from PIL import Image

def mp42np(path):
    cap = cv2.VideoCapture(path)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf = np.empty((frameCount, 224, 224, 3), np.dtype('uint8'))

    fc = 0
    ret = True

    while (fc < frameCount  and ret):
        ret,cur_im = cap.read()
        res_im = np.array(cur_im)
        # print("raw shape:",np.shape(res_im))
        if not len(np.shape(res_im)) == 3:
            print("error during read the file exit this video")
            break
        ## crop at the mid point
        if np.shape(res_im)[0] > np.shape(res_im)[1]:
            mid_point = np.shape(res_im)[0] // 2
            half = np.shape(res_im)[1] // 2
            res_im = res_im[mid_point - half:mid_point + half,:,:]

        elif np.shape(res_im)[0] < np.shape(res_im)[1]:
            mid_point = np.shape(res_im)[1] // 2
            half = np.shape(res_im)[0] // 2    
            res_im = res_im[:,mid_point - half:mid_point + half,:]
        
        # res_im = Image.fromarray(res_im).convert('RGB')
        # resize
        res_im = cv2.resize(res_im, (224,224), interpolation=cv2.INTER_AREA)
        res_im = np.array(res_im)
        # print(np.shape(res_im))
        
        buf[fc] = res_im
        fc += 1

    cap.release()

#     cv2.namedWindow('frame 10')
#    cv2.imshow('frame 10', buf[9])

#     cv2.waitKey(0)
        ## preprocess to make sure that the size is 224*224
    #res_im = np.uint8(np.array(buf))
    #print(np.shape(res_im),"max",np.max(res_im),"min",np.min(res_im))
    # print()

    print("final shape",np.shape(buf))
    
    return np.array(buf)


for folder_name in argv[1:]:
    paths = glob.glob('../data/'+folder_name+'/noFight/*.mp4') + glob.glob('../data/'+folder_name+'/noFight/*.avi')
    paths.sort()
    print("reading from:"+folder_name)
    print("fight_len:",len(paths))

    # dest_path = './data'
    # print(paths)
    for i,path in enumerate(paths): #  r path in paths:
        cur_numpy = mp42np(path)
        dest_path = '../data/spa_npy/'+folder_name+'/noFight/'
        print(dest_path+str(i)+'.npy')
        if not os.path.exists(dest_path):
            print("creating file path")
            os.makedirs(dest_path)
        else: 
            print("file path already exist")
        np.save(dest_path+str(i)+'.npy', cur_numpy)
    
    
    paths = glob.glob('../data/'+folder_name+'/fight/*.mp4') + glob.glob('../data/'+folder_name+'/fight/*.avi')
    paths.sort()
    print("nofight_len:",len(paths)) 
    # dest_path = './data'
    # print(paths)
    for i,path in enumerate(paths):
        print(i)
        cur_numpy = mp42np(path)
        dest_path = '../data/spa_npy/'+folder_name+'/fight/'
        print(dest_path+str(i)+'.npy')
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        np.save(dest_path+str(i)+'.npy',cur_numpy)



