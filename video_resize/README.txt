VTrans
Author: Jinsong Yuan
Descripytion: A video transverter used for Aggression-Detect. Being able to resize and cut videos.
Update Log: first version

Commands:
python vtrans.py [commands]
-h/--help: show the help information about vtrans
-a/-all: transform a batch of videos in a given directory (which can be defined using -i). The format of outputs will be ".mp4" while the original name of file remains. Default paths for input and output are ".\input_dir" and ".\output_dir".
-i: 1 argument [-i path]. Define the input path. If "-all" is used, the path should be directory while file if not.
-o: the same as -i but define the output path.
-s: 2 arguments [-s width height]. Define the resolution ratio for output videos. The default size is 224*224. width and height should be in range of [0, 2000]
-t: 2 arguments [-t start end]. Set time for start and end time (in second) for video cutting. Pay attention that end should be larger than start and both of them are non-negative integers. The end time should no larger than the total time of the original video. If "-all" is used, "-t" cannot be set.

Example:
python vtrans.py -a -s 300 300
python vtrans.py -s 100 100 -t 0 60 -i .\input_dir\1.mp4 .\output_dir\1.mp4