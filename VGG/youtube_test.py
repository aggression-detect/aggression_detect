import os
import numpy as np
from VGG import npy_generator
from VGG import VGG_Model
from VGG import preprocess
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

data_kind_list = ["Movie Dataset", "HockeyFights","SC_delete_MP4","uniform_youtube_MP4","Real_Life_Violence_Dataset","crowd_violence"]
data_kind = data_kind_list[1]
data_youtube = data_kind_list[3]
L = 5
path = os.getcwd()
h5_path = path + "/VGG/weights/weights_"+str(L)+"_"+data_kind+".best.hdf5"
youtube_file, _,_ = preprocess(path, USE_NPY=True, RUN_ALL = False, data_kind = data_youtube, data_list= data_kind_list, L = L, rate = 1, rate2=1)
youtube_generator = npy_generator (youtube_file,1,L)
model = VGG_Model(L=5)
model.load_weights(h5_path)
model.summary()
model.test(youtube_generator,len(youtube_file))