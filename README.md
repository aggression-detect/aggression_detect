# aggression_detect
## team members:  
    Hangzheng Lin  
    Jinsong Yuan  
    Ruike Zhu  
    Xiyue Zhu  


## dataset:
Here we use four  data sets:  
1. Movie - [Link](https://academictorrents.com/details/70e0794e2292fc051a13f05ea6f5b6c16f3d3635)
2. HockeyFights - [Link](https://academictorrents.com/details/38d9ed996a5a75a039b84cf8a137be794e7cee89)
3. Surveillance Camera Fight Dataset - [Link](https://github.com/sayibet/fight-detection-surv-dataset)
4. youtube_fight - [Link](https://github.com/aggression-detect/aggression\_detect)

for each data sets, we resize the video into 224x224 and convert the mp4 into jpg form.

## How to run
Most of the code are run on the Load Sharing Facility (LSF). To upload your job, you can use a script to submit the code.  
For instance, a VGG.sh
```sh
#BSUB -q gpuq
#BSUB -o VGG/output/HK_5_30_0.0001.out
#BSUB -e VGG/output/HK_5_30_0.0001.err
#BSUB -n 2
#BSUB -R "select[ngpus>0] rusage[ngpus_shared=24]"
#BSUB -R span[ptile=2]
#BSUB -a python
 
CURDIR=$PWD
cd $CURDIR
python VGG/VGG.py
```
Submit the LSF job.
```
bsub < VGG.sh
```  
If you are not using LSF, you need to change the path and specify the GPU.  

### Preprocess
1. Download datasets from ______  
2. Use video_resize to resize all the video    
3. Put all the video data into video_image folder.  
4. Run the videos_to_images&opflow.py in data preprocess folder to convert videos to images and optical flow.  
5. Run the opflow_to_npy.py to store optical flow images into numpy array.  

### VGG 
```python  
data_kind_list = ["Movie Dataset", "HockeyFights","SC_delete_MP4","uniform_youtube_MP4","Real_Life_Violence_Dataset","crowd_violence"]  
data_kind = data_kind_list[1]  
```  
Th data_kind_list lists all the video type, change the data_kind to run a specific dataset.  

The code will store the best model into the VGG/weights. Then, you can run the youtube_test.py to load the corresponding checkpoints and test this model on the yutube dataset.

### LSTM+CNN
The original training samples is stored in source folder, and the preprocessed samples are stored in data/raw_frames folder. The pre-load model can be changed by changing the value of cnns_arch variable in run_movie_predict.py.

To train the model and save the .h5 model file, run the below command:
```python  
python run_movie_predict.py 
```  
The model will be saved under the LSTM+CNN folder. 
