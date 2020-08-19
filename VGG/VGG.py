# -*- coding: utf-8 -*-
from math import floor
from sklearn.metrics import classification_report
import numpy as np
import glob
#import cv2
import re
from numpy.random import seed

import os
import h5py
import gc
import re
import tensorflow as tf
from tensorflow.keras.activations import sigmoid 
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Flatten,
		 	  Activation, Dense, Dropout, ZeroPadding2D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization  
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
#from tensorflow.keras import backend as o
#import tensorflow as tf
#import tensorflow.keras.backend.tensorflow_backend as KTF

#os.environ['CUDA_VISIBLE_DEVICES'] = "0"
#config = tf.compat.v1.ConfigProto()
#sess = tf.compat.v1.Session(config=config)
#KTF.set_session(sess)
#K.tensorflow_backend._get_available_gpus()

'''
@USE_NPY: whether to use the .npy file to train
@RUN_ALL: whether to train on all dataset
@data_kind: the training data kind, only will be used if RUN_ALL is False
@data_list: contains all the data kind, only will be used if RUN_ALL is True
@rate: the propotion between training dataset and others
@rate2: the propotion between validation dataset and test dataset
'''


def preprocess(path, USE_NPY=True, RUN_ALL = True, data_kind = None, data_list= None, rate = 0.7, rate2=0.5):
    if not RUN_ALL:
        if USE_NPY:
            data_path = path + "/npy_" + str(L)+ "/" + data_kind + " opflow" 
            print("fight data = ", len(glob.glob(r"%s"%(data_path +"/fight/*.npy"))))
            print("noFight data = ", len(glob.glob(r"%s"%(data_path +"/noFight/*.npy"))))
        else:
            data_path = path + "/video_image"+ "/"+ data_kind + " opflow"
            print ("fight data = ",len(glob.glob(r"%s"%(data_path+"/fight/*.mp4"))))
            print ("noFight data = ",len(glob.glob(r"%s"%(data_path+"/noFight/*.mp4"))))
        print ('data_path = ',data_path)
    else:
        data_path_list = []
        fight_len_list = []
        noFight_len_list = []
        if USE_NPY:
            for data_kind in data_kind_list:
                data_path_list.append(path+ "/npy_"+str(L)+"/"+data_kind + " opflow")
            for data_path in data_path_list:
                fight_len_list.append(len(glob.glob(r"%s"%(data_path+"/fight/*.npy"))))
                noFight_len_list.append(len(glob.glob(r"%s"%(data_path+"/noFight/*.npy"))))
        else:
            for data_kind in data_kind_list:
                data_path_list.append(path + "/video_image"+ "/"+ data_kind + " opflow")
            for data_path in data_path_list:
                fight_len_list.append(len(glob.glob(r"%s"%(data_path+"/fight/*.mp4"))))
                noFight_len_list.append(len(glob.glob(r"%s"%(data_path+"/noFight/*.mp4"))))
        print ("run all")
        print ("fight data = ", sum(fight_len_list))
        print ("noFight data = ", sum(noFight_len_list))
    
    fight_files = []
    noFight_files = []
    if not RUN_ALL:
        if USE_NPY:
            fight_files = glob.glob(data_path+'/fight'+'/*npy')
            noFight_files = glob.glob(data_path+'/noFight'+'/*npy')
        else:
            fight_files = glob.glob(data_path+'/fight'+'/*mp4')
            noFight_files = glob.glob(data_path+'/noFight'+'/*mp4')
        np.random.shuffle(fight_files)
        np.random.shuffle(noFight_files)
        test_file = fight_files[floor(len(fight_files)*rate):]+ noFight_files[floor(len(noFight_files)*rate):]
        train_file = fight_files[:floor(len(fight_files)*rate)]+ noFight_files[:floor(len(noFight_files)*rate)]
    else:
        test_file = []
        valid_file = []
        train_file = []

        if USE_NPY:
            for data_path in data_path_list:
                fight_files += glob.glob(data_path+'/fight'+'/*npy')
                noFight_files += glob.glob(data_path+'/noFight'+'/*npy')
                np.random.shuffle(fight_files)
                np.random.shuffle(noFight_files)
                test_file += fight_files[floor(len(fight_files)*rate):]+ noFight_files[floor(len(noFight_files)*rate):]
                train_file += fight_files[:floor(len(fight_files)*rate)]+ noFight_files[:floor(len(noFight_files)*rate)]
        else:
            for data_path in data_path_list:
                fight_files += glob.glob(data_path+'/fight'+'/*mp4')
                noFight_files += glob.glob(data_path+'/noFight'+'/*mp4')
                np.random.shuffle(fight_files)
                np.random.shuffle(noFight_files)
                test_file += fight_files[floor(len(fight_files)*rate):]+ noFight_files[floor(len(noFight_files)*rate):]
                train_file += fight_files[:floor(len(fight_files)*rate)]+ noFight_files[:floor(len(noFight_files)*rate)]


    np.random.shuffle(test_file)
    np.random.shuffle(train_file)
				
    valid_file = test_file[floor(len(test_file)*rate2):]
    test_file = test_file[:floor(len(test_file)*rate2)]

    print ("test data length = ",len(test_file))
    print ("train data length = ", len(train_file))
    print ("valid data length = ", len(valid_file))
    return train_file, valid_file, test_file 

def get_y (file):
    y_test = []
    print (file)
    for video_path in file:
        flow = np.load(file = video_path)
        if "noFight" in video_path:
            y_test += [0 for i in range(len(flow))]
        else:
            y_test += [1 for i in range(len(flow))]
    
    return y_test


def generator(list1, lits2):
    '''
    Auxiliar generator: returns the ith element of both given list with
         each call to next() 
    '''
    for x,y in zip(list1,lits2):
        yield x, y


def sort_key(s):
    if s:
        try:
            c = re.findall(r"\d+", s)[-1]

        except:
            c = -1
        return int(c)


def data_generator (train_file,batch_size):
    count = 0
    trainx = np.zeros(shape=(0,224,224,2*L), dtype=np.float64)
    trainy = []
    while (1):
        for video_path in train_file:
            count +=1
            x_images = glob.glob(video_path + '/x_*.jpg')
            x_images.sort(key=sort_key)
            y_images = glob.glob(video_path + '/y_*.jpg')
            y_images.sort(key=sort_key)
            nb_stacks = len(x_images)-L+1
            flow = np.zeros(shape=(224,224,2*L,nb_stacks), dtype=np.float64)
            gen = generator(x_images,y_images)
            # print(video_path)
            # print(np.shape(x_images),np.shape(y_images))
            for i in range(len(x_images)):
                flow_x_file, flow_y_file = next(gen)
                img_x = cv2.imread(flow_x_file, cv2.IMREAD_GRAYSCALE)
                img_y = cv2.imread(flow_y_file, cv2.IMREAD_GRAYSCALE)
                # Assign an image i to the jth stack in the kth position, but also
                # in the j+1th stack in the k+1th position and so on
                # (for sliding window) 
                for s in list(reversed(range(min(L,i+1)))):
                    if i-s < nb_stacks:
                        flow[:,:,2*s,i-s] = img_x
                        flow[:,:,2*s+1,i-s] = img_y
                del img_x,img_y
                gc.collect()
            flow = np.transpose(flow, (3, 0, 1, 2))
            # print (np.shape(flow))
            trainx = np.concatenate((trainx, flow), axis=0)
            if "noFight" in video_path:
                trainy += [0 for i in range(len(flow))]
            else:
                trainy += [1 for i in range(len(flow))]

            del flow
            gc.collect()

            if count % batch_size == 0:
                trainy =np.array(trainy)
                trainy.reshape((-1,1))
                yield trainx,trainy
                del trainx,trainy
                gc.collect()
                count = 0
                trainx = np.zeros(shape=(0,224,224,2*L), dtype=np.float64)
                trainy = []

def npy_generator(train_file,batch_size):
    count = 0
    trainx = np.zeros(shape=(0,224,224,2*L), dtype=np.float64)
    trainy = []
    while (1):
        for video_path in train_file:
            count +=1
            flow = np.load(file = video_path)
            trainx = np.concatenate((trainx,flow), axis=0)
            	        
            if "noFight" in video_path:
                trainy += [0 for i in range(len(flow))]
            else:
                trainy += [1 for i in range(len(flow))]
			
            del flow
            gc.collect()
              
            if count % batch_size == 0:
                trainy = np.array(trainy)
                trainy.reshape((-1,1))
                yield trainx,trainy
                del trainx,trainy
                gc.collect()
                count = 0
                trainx = np.zeros(shape=(0,224,224,2*L), dtype=np.float64)
                trainy = []

class VGG_Model():
    def __init__(self,learning_rate = 0.0001, num_features = 4096, L = 10 ):
        self.model = Sequential()

        self.model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 2*L)))
        self.model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_1'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_2'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(128, (3, 3), activation='relu', name='conv2_1'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(128, (3, 3), activation='relu', name='conv2_2'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_1'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_2'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_3'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_1'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_2'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_3'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_1'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_2'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_3'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(4096, name='fc6', kernel_initializer='glorot_uniform'))

        self.model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(4096, name='fc2', kernel_initializer='glorot_uniform'))
        self.model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1, name='predictions',kernel_initializer='glorot_uniform'))
        self.model.add(Activation('sigmoid'))

        adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self.model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    
    def summary(self):
        self.model.summary()   # plot the structure of VGG model
    
    def get_model(self):
        return self.model
    
    def train(self,train_generator,valid_generator,train_file, valid_file,  batch_size, callbacks_list, epoch = 50, verbose=2):
        self.model.fit(train_generator, steps_per_epoch=len(train_file)//batch_size,
				epochs = epoch, validation_data=valid_generator,validation_steps= len(valid_file),
				callbacks = callbacks_list, verbose = 2 )
    
    def load_weights(self, h5_path):
        self.model.load_weights(r'%s'%h5_path) 
    
    def test(self,test_generator,steps,verbose=2):
        self.model.evaluate_generator(test_generator,steps=steps,verbose=verbose)


if __name__ == '__main__':
    # get the work path
    sess = tf.InteractiveSession()
    path = os.getcwd()
    #path = os.path.abspath(os.path.dirname(path)+os.path.sep+".")
    os.chdir(path)
    os.listdir(path)
    print (path) 

    # name list of all dataset 
    data_kind_list = ["Movie Dataset", "HockeyFights", "Surveillance Camera Fight Dataset", "youtube_fight"]
    data_kind = data_kind_list[3]
    USE_NPY = True # whether to use .npy file
    RUN_ALL = False # whether to train on all dataset
    # L: the frame number per stack, generally we set it to 10, so there
    #    will be 10 x-optical flows and 10 y-optical flows per stack
    L = 5 # 5 or 10 
    batch_size = 4
    epoch = 50
    rate = 0.7
    rate2 = 0.5
    print("L =",L,"batch_size = ", batch_size)

    # save the best weight
    if not RUN_ALL:
        h5_path = path + "/VGG/weights/weights_"+str(L)+"_"+data_kind+".best.hdf5"
    else:
        h5_path = path + "/VGG/weights/weights_"+str(L)+"_All.best.hdf5"
    checkpoint = ModelCheckpoint(h5_path, monitor='val_acc', verbose=1, save_best_only=True,mode='max')
    callbacks_list = [checkpoint]
    train_file, valid_file, test_file  = preprocess(path, USE_NPY=USE_NPY, RUN_ALL = RUN_ALL, data_kind = data_kind, data_list= data_kind_list, rate = rate, rate2=rate2)
    model = VGG_Model(L=L)
    model.summary()

    if not USE_NPY:
        train_generator = data_generator (train_file,batch_size)
        valid_generator = data_generator (valid_file,1)
        test_generator = data_generator (test_file,1)
    else:
        train_generator = npy_generator (train_file,batch_size)
        valid_generator = npy_generator (valid_file,1)
        test_generator = npy_generator (test_file, 1)
    
    #model.train(train_generator,valid_generator, train_file, 
    #           valid_file, batch_size=batch_size, callbacks_list=callbacks_list, epoch = epoch, verbose=2)
    
    model.load_weights(h5_path)

    print ('###################### Test ######################')
    model.test(test_generator, len(test_file))


