import os, sys

# define available formats for input videos
in_format = [".flv", ".avi", ".mp4"]

def setFormat(data):
	st = ""
	if (data<10):
		st = "0"+str(data)
	else:
		st = str(data)
	return st

# function to resize all videos (read from input_dir, output to output_dir)
def resize_all(input_dir, output_dir, size):

	# list files to find input videos
	files = os.listdir(input_dir)
	for file in files:
		# split the fname and fextension
		fname, fextension = os.path.splitext(file)

		# filter video files
		if (fextension not in in_format):
			continue

		# read and reszie the video
		in_file_dir = input_dir + "\\" + file
		out_file_dir = output_dir + "\\" + fname + ".mp4"
		os.system("ffmpeg -y -i %s -vf scale=%d:%d %s"% (in_file_dir, size[0], size[1], out_file_dir))

def resize(in_file_dir, out_file_dir, size, start, end):
	if (start==-1):
		os.system("ffmpeg -y -i %s -vf scale=%d:%d %s"% (in_file_dir, size[0], size[1], out_file_dir))
		return
	if (end<start):
		print("Time Error")
		return
	end = end-start
	s = setFormat(start//3600)+":"+setFormat((start%3600)//60)+":"+setFormat(start%60)
	e = setFormat(end//3600)+":"+setFormat((end%3600)//60)+":"+setFormat(end%60) 
	os.system("ffmpeg -y -ss %s -t %s -i %s -vf scale=%d:%d %s"% (s, e, in_file_dir, size[0], size[1], out_file_dir))

args = sys.argv[1:]
arg_dict = {
	"all": False,
	"single": True,
	"width": 224,
	"height": 224,
	"input_dir": ".\\input_dir",
	"output_dir": ".\\output_dir",
	"start": -1,
	"end": -1
}

l = len(args)
i = 0
while (i < l):
	if (args[i]=="-h" or args[i]=="--help"):
		if (l==1):
			pass
			print("-h/--help: show the help")
			print("-a/-all: convert all videos in given directory")
			print("-i: the path of input directory(-all)/file(no -all); default: '.\\input_dir'; [-i path]")
			print("-o: the path of output directory(-all)/file(no -all); defalut: '.\\output_dir'; [-o path]")
			print("-s: set width and height of output video; default: '224 224'; [-s width height]")
			print("-t: set start time and end time for cutting in seconds (cannot use for -all); [-t start end]")
			print("Example 1: python *.py -all -s 300 300")
			print("Example 2: python *.py -s 100 100 -t 0 60 -i .\\input_dir\\1.mp4 -o .\\output_dir\\1.mp4")
			print()
			print("For more information, please visit README file.")
			exit()
		else:
			print("Arguments Error")
			exit()

	elif (args[i]=="-a" or args[i]=="-all"):
		arg_dict["all"] = True
		arg_dict["single"] = False

	elif (args[i]=="-i"):
		arg_dict["input_dir"] = args[i+1]
		if (i+1>=l):
			print("Arguments Error")
			exit()
		i = i + 1

	elif (args[i]=="-o"):
		arg_dict["output_dir"] = args[i+1]
		if (i+1>=l):
			print("Arguments Error")
			exit()
		i = i + 1

	elif (args[i]=="-t"):
		arg_dict["start"] = int(args[i+1])
		if (arg_dict["start"] < 0):
			print("Time Error")
			exit()
		arg_dict["end"] = int(args[i+2])
		if (arg_dict["end"] < 0):
			print("Time Error")
			exit()
		if (i+2>=l):
			print("Arguments Error")
			exit()
		i = i + 2

	elif (args[i]=="-s"):
		arg_dict["width"] = int(args[i+1])
		arg_dict["height"] = int(args[i+2])
		if (arg_dict["width"]<0 or arg_dict["height"]<0 or arg_dict["width"]>2000 or arg_dict["height"]>2000):
			print("Size Error")
			exit()
		if (i+2>=l):
			print("Arguments Error")
			exit()
		i = i + 2;

	else:
		print("Arguments Error")
		exit()

	i = i + 1

if (arg_dict["all"] and arg_dict["start"]>=0):
	print("Cannot cut all videos.")
	exit()

if (arg_dict["all"]):
	resize_all(arg_dict["input_dir"], arg_dict["output_dir"], (arg_dict["width"], arg_dict["height"]))
elif (arg_dict["single"]):
	resize(arg_dict["input_dir"], arg_dict["output_dir"], (arg_dict["width"], arg_dict["height"]), arg_dict["start"], arg_dict["end"])