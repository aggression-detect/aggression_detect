import cv2
import re
import os
import glob
import numpy as np
import imageio
from PIL import Image
from tqdm import tqdm

# create two list for storing the name of fight and no fight videos
fiList=[]
noList=[]
totalList=[]
video_list = ["SC_nodelete_MP4","uniform_SC_MP4","SC_delete_MP4","uniform_youtube_MP4","Real_Life_Violence_Dataset"]
# video_kind = video_list[4]
video_kind = "crowd_violence"
# videos_path = '../../xiyuezhu/xiyue/data/'+video_kind
videos_path = 'video/' + video_kind
for video in os.listdir(videos_path+'/fight'):
    if '.mp4' not in video and '.avi' not in video:
        continue
    fiList.append(video)
for video in os.listdir(videos_path+'/noFight'):
    if '.mp4' not in video and '.avi' not in video:
        continue
    noList.append(video)
totalList = fiList+noList
print('Number of fight videos= ',len(fiList))
print('Number of no fight videos= ',len(noList))
print('Total= ', len(totalList))
print (fiList[0],noList[0])

#The rate of extracting the video frame, here we set it for 1 frame
frameFrequency=1

def video_to_images():
    # loop for fighting videos
    for sourceFileName in tqdm(fiList):
        times=0
        
        # Set the output directory
        outPutDirName='video_image/'+video_kind+'/fight/'+sourceFileName+'/'
        if not os.path.exists(outPutDirName):
            # If it doesn't exit, create it
            os.makedirs(outPutDirName)

        # get the path for each single video
        video_path = os.path.join(videos_path+'/fight', sourceFileName)

        camera = cv2.VideoCapture(video_path)
        
        while True:
            times+=1
            res, image = camera.read()
            # if the video ends, break the loop
            if not res:
                break
            image = cv2.resize(image,(224,224))
            if times > 1 and np.sum((image - last_image)**2) <200000:
                times -=1
                continue
                
            
            if times%frameFrequency==0:
    #             print (np.sum((image - last_image)**2))
                cv2.imwrite(outPutDirName + str(int(times/frameFrequency))+'.jpg', image)
            last_image = image
        camera.release()

    # loop for none-fighting videos
    for sourceFileName in tqdm(noList):
        times=0
        
        # Set the output directory
        outPutDirName='video_image/'+video_kind+'/noFight/'+sourceFileName+'/'
        if not os.path.exists(outPutDirName):
            # If it doesn't exit, create it
            os.makedirs(outPutDirName)

        # get the path for each single video
        video_path = os.path.join(videos_path+'/noFight', sourceFileName)

        camera = cv2.VideoCapture(video_path)
        while True:
            times+=1
            res, image = camera.read()
            
            # if the video ends, break the loop
            if not res:
                break
            image = cv2.resize(image,(224,224))
            if times%frameFrequency==0:
                cv2.imwrite(outPutDirName + str(int(times/frameFrequency))+'.jpg', image)
    print('process finished')

def sort_key(s):
    if s:
        try:
            c = re.findall('^\d+', s)[0]
        except:
            c = -1
        return int(c)

video_to_images()

for sourceFileName in tqdm(fiList):
    # Set the output directory
    outPutDirName='video_image/'+video_kind+' opflow'+'/fight/'+sourceFileName+'/'
    if not os.path.exists(outPutDirName):
        # If it doesn't exit, create it
        os.makedirs(outPutDirName)

    # get the path for each single images set
    image_paths = 'video_image/'+video_kind+'/fight/'+sourceFileName+'/'

    # contains the names of all the images
    imageList = os.listdir(image_paths)
    imageList.sort(key = sort_key)

    # image1 initially stores the first image
    image1 = imageio.imread(image_paths + imageList[0])
    image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
    for image_idx in range(1,len(imageList)):
        image2 = imageio.imread(image_paths + imageList[image_idx])
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        optical_flow = cv2.DualTVL1OpticalFlow_create()
        flow = optical_flow.calc(image1, image2, None)
        opflowx=cv2.normalize(flow[:,:,0], None, 0, 255, cv2.NORM_MINMAX)
        opflowy=cv2.normalize(flow[:,:,1], None, 0, 255, cv2.NORM_MINMAX)
        opflowx = np.array([opflowx]*3).transpose(1,2,0)
        opflowy = np.array([opflowy]*3).transpose(1,2,0)
        cv2.imwrite(outPutDirName + 'x_'+str(image_idx)+'.jpg', opflowx)
        cv2.imwrite(outPutDirName + 'y_'+str(image_idx)+'.jpg', opflowy)
        image1 = image2
print('fight process finished')

for sourceFileName in tqdm(noList):
    # Set the output directory
    outPutDirName='video_image/'+video_kind+' opflow'+'/noFight/'+sourceFileName+'/'
    if not os.path.exists(outPutDirName):
        # If it doesn't exit, create it
        os.makedirs(outPutDirName)

    # get the path for each single images set
    image_paths = 'video_image/'+video_kind+'/noFight/'+sourceFileName+'/'

    # contains the names of all the images
    try:
        imageList = os.listdir(image_paths)
    except:
        continue
    imageList.sort(key = sort_key)
    
    # image1 initially stores the first image
    image1 = imageio.imread(image_paths + imageList[0])
    image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
    for image_idx in range(1,len(imageList)):
        image2 = imageio.imread(image_paths + imageList[image_idx])
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        
        optical_flow = cv2.DualTVL1OpticalFlow_create()
        flow = optical_flow.calc(image1, image2, None)
        opflowx=cv2.normalize(flow[:,:,0], None, 0, 255, cv2.NORM_MINMAX)
        opflowy=cv2.normalize(flow[:,:,1], None, 0, 255, cv2.NORM_MINMAX)
        opflowx = np.array([opflowx]*3).transpose(1,2,0)
        opflowy = np.array([opflowy]*3).transpose(1,2,0)
        cv2.imwrite(outPutDirName + 'x_'+str(image_idx)+'.jpg', opflowx)
        cv2.imwrite(outPutDirName + 'y_'+str(image_idx)+'.jpg', opflowy)
        image1 = image2
        
print('noFight process finished')