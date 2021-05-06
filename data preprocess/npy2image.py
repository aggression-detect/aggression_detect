
import glob
import cv2
import numpy as np
import os
import re
import gc
from tqdm import tqdm
import scipy
import scipy.misc
import matplotlib.pyplot as plt

def sort_key(s):
    if s:
        try:
            c = re.findall(r"\d+", s)[-1]

        except:
            c = -1
        return int(c)

def npy_png(file_dir, dest_dir):
    # 如果不存在对应文件，则创建对应文件
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    file = file_dir
    con_arr = np.load(file)  # 读取npy文件
    for i in range(0, 200):  # 循环数组 最大值为图片张数（我的是200张）  三维数组分别是：图片张数  水平尺寸  垂直尺寸
        try:
            arr = con_arr[i, :, :]  # 获得第i张的单一数组
            disp_to_img = scipy.misc.imresize(arr, [224, 224])  # 根据需要的尺寸进行修改
            plt.imsave(os.path.join(dest_dir, "{}.png".format(i)), disp_to_img, cmap='plasma')  # 定义命名规则，保存图片为彩色模式
            print('photo {} finished'.format(i))
        except:
            break


fiList=[]
noList=[]
totalList=[]
video_list = ["SC_nodelete_MP4","uniform_SC_MP4","SC_delete_MP4","uniform_youtube_MP4","Real_Life_Violence_Dataset"]
video_kind = video_list[4]
videos_path = '../../xiyuezhu/xiyue/data/spa_npy/'+video_kind

fight_files = glob.glob(videos_path+'/fight'+'/*npy')
noFight_files = glob.glob(videos_path+'/noFight'+'/*npy')
fight_files.sort(key=sort_key)
noFight_files.sort(key=sort_key)

i = 0
for file_dir in fight_files:
    dest_dir = "video_image/"+ video_kind +"/fight/" + str(i)
    i+=1
    npy_png(file_dir,dest_dir)

i = 0

for file_dir in noFight_files:
    dest_dir = "video_image/"+ video_kind + "/noFight/" + str(i)
    i+=1
    npy_png(file_dir,dest_dir)




