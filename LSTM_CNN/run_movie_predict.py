import os
from itertools import chain
import cv2

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.optimizers import RMSprop, Adam

from tensorflow.keras.applications import Xception, ResNet50, InceptionV3, MobileNet, VGG19, DenseNet121, InceptionResNetV2, VGG16
from tensorflow.keras.layers import LSTM, ConvLSTM2D
import BuildModel_basic
import DatasetBuilder
from tensorflow.keras.callbacks import ModelCheckpoint
#from tensorflow.keras.model import load_weights
import tensorflow

import pandas as pd
from numpy.random import seed, shuffle

# from tensorflow import set_random_seed
from collections import defaultdict
import GPUtil
import numpy as np

TIME_INTERVAL = 4


first_called = 1
result = 0
model = 0
res_scores = []

def predict_main():
    class TestCallback(Callback):
        def __init__(self, test_data):
            self.test_data = test_data
            self.test_loss = []
            self.test_acc = []

        def on_epoch_end(self, epoch, logs={}):
            x, y = self.test_data
            loss, acc = self.model.evaluate(x, y, batch_size=10, verbose=0)
            self.test_loss.append(loss)
            self.test_acc.append(acc)
            GPUtil.showUtilization()
            print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))


    def train_eval_network(dataset_name, train_gen, validate_gen, test_x, test_y, seq_len, epochs, batch_size,
                        batch_epoch_ratio, initial_weights, size, cnn_arch, learning_rate,
                        optimizer, cnn_train_type, pre_weights, lstm_conf, len_train, len_valid, dropout, classes,
                        patience_es=15, patience_lr=5):
        """the function build, compine fit and evaluate a certain architechtures on a dataset"""
        global first_called
        global result
        global model
        global ave_acore
        if first_called:
            print("loading the model")
            #tensorflow.set_random_seed(2)
            tensorflow.random.set_random_seed(2)
            seed(1)
            result = dict(dataset=dataset_name, cnn_train=cnn_train_type,
                        cnn=cnn_arch.__name__, lstm=lstm_conf[0].__name__, epochs=epochs,
                        learning_rate=learning_rate, batch_size=batch_size, dropout=dropout,
                        optimizer=optimizer[0].__name__, initial_weights=initial_weights, seq_len=seq_len)
            print("run experimnt " + str(result))
            model = BuildModel_basic.build(size=size, seq_len=seq_len, learning_rate=learning_rate,
                                        optimizer_class=optimizer, initial_weights=initial_weights,
                                        cnn_class=cnn_arch, pre_weights=pre_weights, lstm_conf=lstm_conf,
                                        cnn_train_type=cnn_train_type, dropout=dropout, classes=classes)

            # the network is trained on data generatores and apply the callacks when the validation loss is not improving:
            # 1. early stop to training after n iteration
            # 2. reducing the learning rate after k iteration where k< n
            test_history = TestCallback((test_x, test_y))
            #model_path = './model_16_.h5'
            #checkpoint = ModelCheckpoint(filepath=model_path,monitor='val_loss',mode='auto' ,save_best_only='True')

            #default_values = dict(epoch=10,\
            #                  learning_rate=0.0004,\
            #                  batch_size=16,\
            #                  optimizer=Adam,\
            #                  initial_weights=0,\
            #                  cnn_class=Xception,\
            #                  pre_weights='Xavier',\
            #                  lstm_conf=(LSTM,dict(units = 256)),\
            #                  cnn_train_type='static'
            #                  )
            #model.compile(optimizer=default_values["optimizer"], loss='categorical_crossentropy')
            
            h5model_path = './model_sc_2g_transfer.h5'
            model.load_weights(h5model_path)
            first_called = 0
        # print("predicting..")
        history = model.predict_generator(
            validate_gen,
            int(float(len_valid) / float(batch_size))
        )
        
        # print(history)
        ave_acore = np.sum(history)/len(history)
        print(ave_acore)
        res_scores.append(ave_acore)
        # print(model.metrics_names)
        return 1
        history_to_save = history.history
        history_to_save['test accuracy'] = test_history.test_acc
        history_to_save['test loss'] = test_history.test_loss

        model_name = ""
        for k, v in result.items():
            model_name = model_name + "_" + str(k) + "-" + str(v).replace(".", "d")
        model_path = os.path.join(res_path, model_name)
        pd.DataFrame(history_to_save).to_csv(model_path + "_train_results.csv")
        result['validation loss'] = min(history.history['val_loss'])
        result['validation accuracy'] = max(history.history['val_acc'])
        result['last validation loss'] = history.history['val_loss'][-1]
        result['last validation accuracy'] = history.history['val_acc'][-1]

        result['train accuracy'] = max(history.history['acc'])
        result['train loss'] = min(history.history['loss'])
        result['last train accuracy'] = history.history['acc'][-1]
        result['last train loss'] = history.history['loss'][-1]

        result['test accuracy'] = max(test_history.test_acc)
        result['test loss'] = min(test_history.test_loss)
        result['last test accuracy'] = test_history.test_acc[-1]
        result['last test loss'] = test_history.test_loss[-1]

        result['final lr'] = history.history['lr'][-1]
        result['total epochs'] = len(history.history['lr'])
        return result


    def get_generators(dataset_name, dataset_videos, datasets_frames, fix_len, figure_size, force, classes=1, use_aug=False,
                    use_crop=True, crop_dark=None):
        train_path, valid_path, test_path, \
        train_y, valid_y, test_y, \
        avg_length = DatasetBuilder.createDataset(dataset_videos, datasets_frames, fix_len, force=force)

        if fix_len is not None:
            avg_length = fix_len
        crop_x_y = None
        if (crop_dark):
            crop_x_y = crop_dark[dataset_name]

        len_train, len_valid = len(train_path), len(valid_path)
        # print("in main, validate paths:",valid_path)
        # print("in main, validate labels:",valid_y)
        
        train_gen = DatasetBuilder.data_generator(train_path, train_y, batch_size, figure_size, avg_length, use_aug=use_aug,
                                                use_crop=use_crop, crop_x_y=crop_x_y, classes=classes)
        validate_gen = DatasetBuilder.data_generator(valid_path, valid_y, batch_size, figure_size, avg_length,
                                                    use_aug=False, use_crop=False, crop_x_y=crop_x_y, classes=classes)
        # print("testing gen")
        # for X,y in validate_gen():
        #     print(y)
        test_x, test_y = DatasetBuilder.get_sequences(test_path, test_y, figure_size, avg_length, crop_x_y=crop_x_y,
                                                    classes=classes)

        return train_gen, validate_gen, test_x, test_y, avg_length, len_train, len_valid


    def hyper_tune_network(dataset_name, epochs, batch_size, batch_epoch_ratio, figure_size, initial_weights, lstm,
                        cnns_arch,
                        learning_rates, optimizers, cnn_train_types, dropouts, classes, use_augs, fix_lens):
        """ the function train several networks parameters in a loop and select the best architechture to the next evaluation"""
        results = []

        best_accuracy = 0.0
        best_loss = 10.0
        # static params for tunning
        params_to_train = dict(dataset_name=dataset_name, epochs=epochs, batch_size=batch_size,
                            batch_epoch_ratio=batch_epoch_ratio, initial_weights=initial_weights, size=figure_size,
                            pre_weights=weights, lstm_conf=lstm, classes=classes, patience_es=5, patience_lr=3)

        # the tunning is not evaluation all possible combinations
        # given the importance order of the hyperparams, in each iteraction we choose the best performing parmaters
        exp_params_order = ['cnn_arch', 'learning_rate', 'seq_len', 'use_aug', 'dropout',
                            'cnn_train_type']  # 'cnn_arch','learning_rate','fix_len','use_aug','dropout', 'optimizer','optimizer',
        #

        best_params_train = dict(optimizer=optimizers[0], learning_rate=learning_rates[0],
                                cnn_train_type=cnn_train_types[0], cnn_arch=list(cnns_arch.values())[0],
                                dropout=dropouts[0])
        exp_params_train = dict(optimizer=optimizers[1:], learning_rate=learning_rates[1:],
                                cnn_train_type=cnn_train_types[1:], dropout=dropouts[1:],
                                cnn_arch=cnns_arch.values())

        best_params_data = dict(use_aug=use_augs[0], seq_len=fix_lens[0])
        exp_params_data = dict(use_aug=use_augs[1:], seq_len=fix_lens[1:])

        for exp_param in exp_params_order:
            temp_param = dict(best_params_train)
            temp_param_data = dict(best_params_data)
            if exp_param in exp_params_data:
                exp_params_ = exp_params_data
            else:
                exp_params_ = exp_params_train
            for param in exp_params_[exp_param]:
                if exp_param in best_params_data:
                    temp_param_data[exp_param] = param
                else:
                    temp_param[exp_param] = param

                print(temp_param_data)
                print(temp_param)
                params_to_train['train_gen'], params_to_train['validate_gen'], params_to_train['test_x'], \
                params_to_train['test_y'], params_to_train['seq_len'], params_to_train['len_train'], \
                params_to_train['len_valid'] = get_generators(dataset_name, datasets_videos[dataset_name], datasets_frames,
                                                            temp_param_data['seq_len'],
                                                            figure_size, use_aug=temp_param_data['use_aug'], force=force,
                                                            classes=classes)

                params_to_train.update(temp_param)
                result = train_eval_network(**params_to_train)
                result.update(temp_param_data)
                print(result)
                results.append(result)
                if result['test accuracy'] >= best_accuracy: 
                    # and result['test loss'] <= best_loss :
                    best_accuracy = result['test accuracy']
                    best_loss = result['test loss']
                    if exp_param in best_params_data:
                        best_params_data[exp_param] = param
                    else:
                        best_params_train[exp_param] = param
                    print("best accuracy update " + str(best_accuracy))
        best_params_train.update(best_params_data)
        return best_params_train, results


    # static parameter for the netwotk
    datasets_videos = dict(
        hocky=dict(hocky="./source/youtubeFight"),#HockeyFights"), #youtubeFight"),
        #violentflow=dict(violentflow="data/raw_videos/violentflow"),
        #movies=dict(movies="data/raw_videos/movies")
    )
    #print(datasets_videos)
    crop_dark = dict(
        hocky=None,
        violentflow=None,
        movies=None
    )

    datasets_frames = "data/raw_frames" # "data_youtube/raw_frames"
    res_path = "results"
    figure_size = 244
    # split_ratio = 0.1
    batch_size =10#16# 2
    # batch_epoch_ratio = 0.5 #double the size because we use augmentation
    fix_len = 20
    initial_weights = 'glorot_uniform'
    weights = 'imagenet'
    force = True
    lstm = (ConvLSTM2D, dict(filters=256, kernel_size=(3, 3), padding='same', return_sequences=False))
    classes = 1

    # hyper parameters for tunning the network
    cnns_arch = dict(ResNet50=ResNet50)  # , InceptionV3=InceptionV3, VGG19=VGG19
    learning_rates = [1e-40] #, 1e-3
    use_augs = [True] #False
    fix_lens = [10] #20, 
    optimizers = [(RMSprop, {})]#, (Adam, {})]
    dropouts = [0.0] #0.5
    cnn_train_types = ['retrain']#, 'static']

    apply_hyper = False

    if apply_hyper:
        # the hyper tunning symulate the architechture behavior
        # we set the batch_epoch_ratio - reduced by X to have the hypertunning faster with epoches shorter
        hyper, results = hyper_tune_network(dataset_name='hocky', epochs=1,
                                            batch_size=batch_size, batch_epoch_ratio=1, figure_size=figure_size,
                                            initial_weights=initial_weights, lstm=lstm,
                                            cnns_arch=cnns_arch, learning_rates=learning_rates,
                                            optimizers=optimizers, cnn_train_types=cnn_train_types, dropouts=dropouts,
                                            classes=classes, use_augs=use_augs, fix_lens=fix_lens)

        pd.DataFrame(results).to_csv("results_hyper.csv")
        cnn_arch, learning_rate, optimizer, cnn_train_type, dropout, use_aug, fix_len = hyper['cnn_arch'], \
                                                                                        hyper['learning_rate'], \
                                                                                        hyper['optimizer'], \
                                                                                        hyper['cnn_train_type'], \
                                                                                        hyper['dropout'], hyper['use_aug'], \
                                                                                        hyper['seq_len'],
    else:
        results = []
        cnn_arch, learning_rate, optimizer, cnn_train_type, dropout, use_aug, fix_len = ResNet50, 0.0001, (
        RMSprop, {}), 'retrain', 0.0, True, 10

    # apply best architechture on all datasets with more epochs
    for dataset_name, dataset_videos in datasets_videos.items():
        train_gen, validate_gen, test_x, test_y, seq_len, len_train, len_valid = get_generators(dataset_name,
                                                                                                dataset_videos,
                                                                                                datasets_frames, fix_len,
                                                                                                figure_size,
                                                                                                force=force,
                                                                                                classes=classes,
                                                                                                use_aug=use_aug,
                                                                                                use_crop=True,
                                                                                                crop_dark=crop_dark)
        
        
        
        result = train_eval_network(epochs=1, dataset_name=dataset_name, train_gen=train_gen, validate_gen=validate_gen,
                                    test_x=test_x, test_y=test_y, seq_len=seq_len, batch_size=batch_size,
                                    batch_epoch_ratio=0.5, initial_weights=initial_weights, size=figure_size,
                                    cnn_arch=cnn_arch, learning_rate=learning_rate,
                                    optimizer=optimizer, cnn_train_type=cnn_train_type,
                                    pre_weights=weights, lstm_conf=lstm, len_train=len_train, len_valid=len_valid,
                                    dropout=dropout, classes=classes)
        # results.append(result)
        # pd.DataFrame(results).to_csv("results_datasets.csv")
        # print(result)
    # pd.DataFrame(results).to_csv("results.csv")


# for i in range(10):
#     predict_main()
#     print("test continue, current res:")
#     print(res_scores)

def save_mp4(array,target_path):
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    out = cv2.VideoWriter(target_path,fourcc, 24.0, (224,224),isColor=True)
    # print("value of the array  max:", np.max(array), "min:", np.min(array))
    for i in range(len(array)):
        out.write(np.uint8(array[i]))
    out.release()


path = "./movie_yewen/movie.mp4"
cap = cv2.VideoCapture(path)
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fc = 0
ret = True
print("there are totally ",frameCount," frames")
video = []
for fc in range(frameCount): #(fc < frameCount  and ret):
    # print("current frame: ",fc)
    ret,cur_im = cap.read()
    
    # if ret:
    #     print("unknown error. ret is 1 at #"+str(fc)+" frame.")
    #     # break
    res_im = np.array(cur_im)
    # print("raw shape:",np.shape(res_im))
    if not len(np.shape(res_im)) == 3:
        print("error during read the file exit this video")
        break
    ## crop at the mid point
    if np.shape(res_im)[0] > np.shape(res_im)[1]:
        mid_point = np.shape(res_im)[0] // 2
        half = 224*2 #np.shape(res_im)[1] // 2
        res_im = res_im[mid_point - half:mid_point + half,:,:]

    elif np.shape(res_im)[0] < np.shape(res_im)[1]:
        mid_point = np.shape(res_im)[1] // 2
        half =  224*2 #np.shape(res_im)[0] // 2    
        res_im = res_im[:,mid_point - half:mid_point + half,:]
    
    # res_im = Image.fromarray(res_im).convert('RGB')
    # resize
    res_im = cv2.resize(res_im, (224,224), interpolation=cv2.INTER_NEAREST)
    res_im = np.array(res_im)
    #  print(np.shape(res_im))
    
    ## have a buffer to remember the recent  (24*TIME_INTERVAL+10) frames:
    video.append(res_im)
    if len(video) > (24*TIME_INTERVAL+10):
        video.pop(0)

    if fc%(24*TIME_INTERVAL) == 5 and len(video) == (24*TIME_INTERVAL+10):
        print("save video and starting to predict at "+str(fc//(24*TIME_INTERVAL))+" s.")
        ## save the array to mp4 file:
        for i in range(10):
            current_array = video[i:i+(24*TIME_INTERVAL)] 
            target_path = "./source/youtubeFight/fi_"+str(i)+".mp4"       
            save_mp4(current_array,target_path)
        predict_main()
        print("test continue, current res:")
        print(res_scores)        
    

    if fc%(24*TIME_INTERVAL) == 0 and len(video) >= (24*TIME_INTERVAL):# save for human check
        # print("save for human check")
        print("save for human check at "+str(TIME_INTERVAL*fc//(24*TIME_INTERVAL))+" s.")

        current_array = video[-(24*TIME_INTERVAL):]
        target_path = "./human_check/fi_"+str(TIME_INTERVAL*fc//(24*TIME_INTERVAL))+".mp4" 
        save_mp4(current_array,target_path)
        

cap.release()
