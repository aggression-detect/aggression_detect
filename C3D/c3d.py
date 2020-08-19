# -*- coding: utf-8 -*-
from math import floor
import numpy as np
import glob
import cv2
import re
from numpy.random import seed
seed(1)
import os
import h5py
import gc
import re
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import (Input, Conv3D, MaxPooling3D, AveragePooling3D,Flatten,
		 	  Activation, Dense, Dropout, ZeroPadding3D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization  
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# import openvino

def preprocess(path, RUN_ALL = True, data_kind = None, data_list= None, rate = 0.7, rate2=0.5):
    data_path = path+"/video_image"+ "/"+ data_kind
    print ("fight data = ",len(glob.glob(r"%s"%(data_path+"/fight/*.mp4"))))
    print ("noFight data = ",len(glob.glob(r"%s"%(data_path+"/noFight/*.mp4"))))

    fight_files = glob.glob(data_path+'/fight'+'/*mp4')
    noFight_files = glob.glob(data_path+'/noFight'+'/*mp4')

    print ("fight files:", len(fight_files))
    print ("noFight files:",len(noFight_files))
    np.random.shuffle(fight_files)
    np.random.shuffle(noFight_files)


    train_file = fight_files[floor(len(fight_files)*rate):]+ noFight_files[floor(len(noFight_files)*rate):]
    test_file = fight_files[:floor(len(fight_files)*rate)]+ noFight_files[:floor(len(noFight_files)*rate)]
    
    valid_file = test_file[floor(len(test_file)*rate2):]
    test_file = test_file[:floor(len(test_file)*rate2)]
    print ("test data length = ",len(test_file))
    print ("train data length = ", len(train_file))
    print ("valid data length = ", len(valid_file))
    return train_file, valid_file, test_file 

def sort_key(s):
    if s:
        try:
            c = re.findall(r"\d+", s)[-1]

        except:
            c = -1
        return int(c)

def generator(list1, lits2):
    '''
    Auxiliar generator: returns the ith element of both given list with
         each call to next() 
    '''
    for x,y in zip(list1,lits2):
        yield x, y



class C3d_Model():
    def __init__(self,learning_rate = 0.0001, num_features = 4096, L= 16):
        self.model = Sequential()
        
        # reshape the input images from 224*224 into 112*112
        self.model.add(AveragePooling3D((1,2,2),strides=(1,2,2)))
        
        self.model.add(ZeroPadding3D((1, 1, 1), input_shape=(L, 112, 112, 3)))
        self.model.add(Conv3D(64, (3, 3, 3), activation='relu', name='conv1a'))
        self.model.add(MaxPooling3D((1, 2, 2), strides=(1, 2, 2)))

        self.model.add(ZeroPadding3D((1, 1, 1), input_shape=(L, 56, 56, 3)))
        self.model.add(Conv3D(128, (3, 3, 3), activation='relu', name='conv2a'))
        self.model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2)))

        self.model.add(ZeroPadding3D((1, 1, 1), input_shape=(8, 28, 28, 3)))
        self.model.add(Conv3D(256, (3, 3, 3), activation='relu', name='conv3a'))
        self.model.add(ZeroPadding3D((1, 1, 1), input_shape=(8, 28, 28, 3)))
        self.model.add(Conv3D(256, (3, 3, 3), activation='relu', name='conv3b'))
        self.model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2)))


        self.model.add(ZeroPadding3D((1, 1, 1), input_shape=(4, 14, 14, 3)))
        self.model.add(Conv3D(512, (3, 3, 3), activation='relu', name='conv4a'))
        self.model.add(ZeroPadding3D((1, 1, 1), input_shape=(4, 14, 14, 3)))
        self.model.add(Conv3D(512, (3, 3, 3), activation='relu', name='conv4b'))
        self.model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2)))

        self.model.add(ZeroPadding3D((1, 1, 1), input_shape=(2, 7, 7, 3)))
        self.model.add(Conv3D(512, (3, 3, 3), activation='relu', name='conv5a'))
        self.model.add(ZeroPadding3D((1, 1, 1), input_shape=(2, 7, 7, 3)))
        self.model.add(Conv3D(512, (3, 3, 3), activation='relu', name='conv5b'))
        self.model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(num_features, name='fc6', kernel_initializer='glorot_uniform'))

        self.model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.1))

        self.model.add(Dense(num_features, name='fc7', kernel_initializer='glorot_uniform'))
        self.model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(1, name='predictions',kernel_initializer='glorot_uniform'))
        self.model.add(Activation('sigmoid'))

        adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999,
            epsilon=1e-08)
        self.model.compile(optimizer=adam, loss='binary_crossentropy',
            metrics=['accuracy'])
        
        # self.model.build((None,L,224,224,3))
    def summary(self):
        self.model.summary()   # plot the structure of VGG model
    
    def get_model(self):
        return self.model

    def train(self,train_generator,valid_generator,train_file, valid_file,  batch_size, callbacks_list, epoch = 50, verbose=2):
        self.model.fit(train_generator, steps_per_epoch=len(train_file)//batch_size,
				epochs = epoch, validation_data=valid_generator,validation_steps= len(valid_file)//batch_size,
				callbacks = callbacks_list, verbose = 2 )

    def load_weights(self, h5_path):
        self.model.load_weights(r'%s'%h5_path) 
    
    def test(self,test_generator,steps,verbose=2):
        self.model.evaluate_generator(test_generator,steps=steps,verbose=verbose)
             
def data_generator(train_file,batch_size,L=16):
    count = 0
    trainx = np.zeros(shape=(0,L,224,224,3), dtype=np.float64)
    trainy = []
    while (1):
        for video_path in train_file:
            count +=1
            
            images = glob.glob(video_path + '/*.jpg')
            images.sort(key=sort_key)
            nb_stacks = len(images)-L+1
            stack = np.zeros(shape=(224,224,3,L,nb_stacks), dtype=np.float64)
            # print(video_path)
            # print(np.shape(x_images),np.shape(y_images))

            for i in range(len(images)):
                img = cv2.imread(images[i], cv2.IMREAD_COLOR)
                # Assign an image i to the jth stack in the kth position, but also
                # in the j+1th stack in the k+1th position and so on
                # (for sliding window) 
                for s in list(reversed(range(min(L,i+1)))):
                    if i-s < nb_stacks:
                        stack[:,:,:,s,i-s] = img
                del img
                gc.collect()                
            
            stack = np.transpose(stack, (4, 3, 0, 1, 2))
            trainx = np.concatenate((trainx, stack), axis=0)
			
            if "noFight" in video_path:
                trainy += [0 for i in range(len(stack))]
            else:
                trainy += [1 for i in range(len(stack))]
            del stack
            gc.collect()
            if count % batch_size == 0:
                trainy =np.array(trainy)
                trainy.reshape((-1,1))
                yield trainx,trainy
                del trainx,trainy
                gc.collect()
                count = 0
                trainx = np.zeros(shape=(0,L, 224,224,3), dtype=np.float64)
                trainy = []


if __name__ == '__main__':
    # get the work path
    path = os.getcwd() 
    #path = os.path.abspath(os.path.dirname(path)+os.path.sep+".")
    os.chdir(path)
    os.listdir(path)
    print (path)

    # name list of all dataset 
    data_kind_list = ["Movie Dataset", "HockeyFights", "Surveillance Camera Fight Dataset", "youtube_fight"]
    data_kind = data_kind_list[1]
    RUN_ALL = True # whether to train on all dataset
    
    
    # size for potical flow batch
    L = 16
    epoch = 50
    batch_size = 3
    rate = 0.7
    rate2 = 0.5
    print("L =",L,"batch_size = ", batch_size)
    if not RUN_ALL:
        h5_path = path + "/C3D/weights/weights_"+str(L)+"_"+data_kind+".best.hdf5"
    else:
        h5_path = path + "/C3D/weights/weights_"+str(L)+"_ALL.best.hdf5"
    checkpoint = ModelCheckpoint(h5_path, monitor='val_acc', verbose=1, save_best_only=True,mode='max')
    callbacks_list = [checkpoint]
    train_file, valid_file, test_file  = preprocess(path,RUN_ALL = RUN_ALL, data_kind = data_kind, data_list= data_kind_list, rate = rate, rate2=rate2)

    train_generator = data_generator (train_file,batch_size)
    valid_generator = data_generator (valid_file,batch_size)
    test_generator = data_generator (test_file,batch_size)
    
    model = C3d_Model(L =L)
    model.train(train_generator,valid_generator, train_file, 
                valid_file, batch_size=batch_size, callbacks_list=callbacks_list, epoch = epoch, verbose=2)
    model.summary()

    model.load_weights(h5_path)
    print ('###################### Test ######################')
    model.test(test_generator, len(test_file))
