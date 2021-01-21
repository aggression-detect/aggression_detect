from torchvision import transforms
import os
import torch
from PIL import Image

image_transforms = {
	'train':
		transforms.Compose([
#			transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
#			transforms.RandomRotation(degrees=15),
#			transforms.ColorJitter(),
#			transforms.RandomHorizontalFlip(),
			transforms.Resize(size=256),
			transforms.CenterCrop(size=224),  # Image net standards
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406],
								 [0.229, 0.224, 0.225])
		]),
	'valid':
		transforms.Compose([
			transforms.Resize(size=256),
			transforms.CenterCrop(size=224),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406],
								 [0.229, 0.224, 0.225])
		])
}

from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math 
batch_size = 64
learn_rate = 0.0001
momentum = 0.6
n_epochs = 50


class MyDataset(Dataset):
	def __init__(self, data_dir, valid = 0, transform = None, stack_size=20):
		self.label_name = {"Fight": 0, "Nofight": 1}
		self.data_info = self.get_img_info(data_dir, stack_size, valid)
		self.transform = transform
		self.stack_size = stack_size
	
	def __getitem__(self, index):
		'''path_stack, s_index, label = self.data_info[index]
		int_label = self.label_name[label]	
	
		img_list = []
		for image_name_i in range(s_index, s_index+self.stack_size):
			image_name = str(image_name_i)+".jpg"
			path_image = os.path.join(path_stack,image_name)
			img = Image.open(path_image).convert('RGB')
			if self.transform is not None:
				img = self.transform(img)
			img_list.append(img)
		stack = torch.stack(img_list[:self.stack_size], dim=0)
		return stack, int_label'''
		data, label = self.data_info[index]
		if (label=="fight"):
			label = "Fight"
		else:
			label = "Nofight"
		int_label = self.label_name[label]
		stack = torch.from_numpy(data[:self.stack_size])
		return stack, int_label
			
		

	def __len__(self):
		return len(self.data_info)

	@staticmethod
	def get_img_info(data_dir, stack_size, valid):
		'''data_info = list()
		root = data_dir
		for sub_label_dir in os.listdir(data_dir):
			stack_names = os.listdir(os.path.join(root, sub_label_dir))
			for i in range(len(stack_names)):
				stack_name = stack_names[i]
				path_stack = os.path.join(root, sub_label_dir, stack_name)
				label = sub_label_dir
				for i in range(1,35-stack_size+2):
					data_info.append((path_stack, i, label))
		return data_info'''
		data_info = list()
		root = data_dir
		for sub_label_dir in os.listdir(data_dir):
			stack_names = os.listdir(os.path.join(root, sub_label_dir))
			stack_len = len(stack_names)
			stack_len = math.ceil(stack_len*.8)
			if (valid):
				stack_names = stack_names[stack_len:]
			else:
				stack_names = stack_names[:stack_len]
			for i in range(len(stack_names)):
				stack_name = stack_names[i]
				path_stack = os.path.join(root, sub_label_dir, stack_name)
				arr = np.load(path_stack)
				arr = np.swapaxes(arr, 2, 3)
				arr = np.swapaxes(arr, 1, 2)
				#if arr.shape[0]<30:
					#print(sub_label_dir, stack_name, arr.shape)
				label = sub_label_dir
				#print(stack_name, ":", arr.shape)
				l = len(arr)
				for i in range(0, l-stack_size+1):
					data_info.append((arr[i:i+stack_size], label))
		return data_info

s_size = 20

'''data = {
	'train':
		MyDataset(data_dir="/gpfsdata/home/jinsongyuan/ai_datastore/datasource/agg_detect/train/Hockey", transform=image_transforms['train'], stack_size=s_size),
	'valid':
		MyDataset(data_dir="/gpfsdata/home/jinsongyuan/ai_datastore/datasource/agg_detect/valid/Hockey", transform=image_transforms['valid'],stack_size=s_size)
}'''

data = {
	'train':
		MyDataset(data_dir="/gpfsdata/home/xiyuezhu/xiyue/data/spa_npy/Surveillance_Camera_Fight_Dataset", transform=image_transforms['train'], stack_size=s_size, valid = 0),
	'valid':
		MyDataset(data_dir="/gpfsdata/home/xiyuezhu/xiyue/data/spa_npy/Surveillance_Camera_Fight_Dataset", transform=image_transforms['valid'], stack_size=s_size, valid = 1)
}

#for i  in range(16):
#	print(data['train'].data_info[i])
#exit()
#print(len(data['train'].data_info))

dataloaders = {
	'train': DataLoader(data['train'], batch_size=batch_size, shuffle=True, num_workers=4),
	'valid': DataLoader(data['valid'], batch_size=batch_size, shuffle=True, num_workers=4)
}

#trainiter = iter(dataloaders['train'])
#features, labels = next(trainiter)
#print(len(features), len(labels))
#print(features[0].shape,labels[0])

#dataiter = iter(dataloaders['train'])
#stacks, labels = dataiter.next()
#print(stacks.size(), labels.size())

#exit()

#import torchvision.models as models
googlenet = torch.load("../../pretrainmodel/googlenet_imagenet.pkl")
#model = models.googlenet(pretrained=True)

import torch.nn as nn

# model.fc = nn.Linear(model.fc.in_features, 2)
'''model.fc = nn.Sequential(
		   #nn.LSTM(input_size=model.fc.in_features, hidden_size=512, num_layers=1, batch_first=True),
		   nn.Linear(model.fc.in_features, 256),
		   nn.ReLU(),
		   nn.Dropout(0.4),
		   nn.Linear(256, 2),
		   nn.LogSoftmax(dim=1))'''

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, x): 
        return x.view(self.shape)

class LSTMBehind(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class):
        super(LSTMBehind, self).__init__()
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, n_class)
    def forward(self, x): 
        out, (hn, hc) = self.lstm(x, None)
        out = out[:,-1,:]
        out = self.classifier(out)
        return out 

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, input_seq):
#        assert len(input_seq.size()) > 2

        # reshape input data --> (samples * timesteps, input_size)
        # squash timesteps
        reshaped_input = input_seq.contiguous().view(-1, input_seq.size(-3), input_seq.size(-2), input_seq.size(-1))

        output = self.module(reshaped_input)
        # We have to reshape Y
        if self.batch_first:
            # (samples, timesteps, output_size)
            output = output.contiguous().view(input_seq.size(0), -1, output.size(-1))
        else:
            # (timesteps, samples, output_size)
            output = output.contiguous().view(-1, input_seq.size(1), output.size(-1))
        return output

class CombinedModel(nn.Module):
        def __init__(self, out_model):
                super(CombinedModel, self).__init__()
                #CNN1 = nn.Conv2d(in_channels=3, kernel_size=7, stride=2, padding=3, bias=False, out_channels=32)
                #googlenet = torch.load('googlenet_imagenet.pkl')
                self.TD = TimeDistributed(out_model, True)
                self.LSTM = LSTMBehind(1000, 256, 1, 2)
        def forward(self, x):
                x = self.TD(x)
                x = self.LSTM(x)
                return x




#model.fc = nn.Sequential(
#            nn.Linear(1024, 1000, True),
            #nn.Linear(1000, 2, True))
#            Reshape(-1, 20, 50),
#            LSTMBehind(50, 25, 1, 400),
#            nn.Linear(400,2))

#print(model)

#exit()

use_gpu = torch.cuda.is_available()

for param in googlenet.parameters():
	param.requires_grad = False

#for para in model.fc.parameters():
#	para.requires_grad = True


model = CombinedModel(googlenet)

from torch import optim
from tqdm import tqdm

print_info = True
criteration = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learn_rate, betas=(0.9,0.999), eps=1e-8)
if (use_gpu):
	model = model.cuda()
	criteration = criteration.cuda()
print("Train Start")
ans = []
for epoch in range(n_epochs):
	if (print_info): print("Epoch:", epoch)
	#datal = enumerate(dataloaders['train'],0)
	running_loss = 0.0
	running_acc = 0.0
	l = len(dataloaders['train'])
	l2 = len(dataloaders['train'].dataset)
	#l = len(list(datal))
	t = tqdm(dataloaders['train'])
	i = 0
	for data, targets in t:
		i = i+1
		if (use_gpu):
			data = data.cuda()
			targets = targets.cuda()
		out = model(data)
		#print(out.shape)
		loss = criteration(out, targets)
		if (use_gpu):
			_, preds = torch.max(out.cpu(), 1)
		else:
			_, preds = torch.max(out, 1)
		loss.backward()
		optimizer.step()
		running_loss = running_loss + float(loss.item())
		if (use_gpu):
			running_acc += torch.sum(preds==targets.data.cpu())
		else:
			running_acc += torch.sum(preds==targets.data)
		t.set_description("Epoch %d"%epoch)
		t.set_postfix(l=loss.item(), loss=running_loss/l, acc=running_acc/l2)
		if (print_info): print(i,"/",l,'    loss:',running_loss/l,"accuracy:",running_acc/l2)

	running_loss = 0.0
	running_acc = 0.0
	t = tqdm(dataloaders['valid'])
	l = len(dataloaders['valid'])
	l2 = len(dataloaders['valid'].dataset)
	i = 0
	if (print_info): print("Valid:")
	for data, targets in t:
		if (use_gpu):
			data = data.cuda()
			targets = targets.cuda()
		out = model(data)
		loss = criteration(out, targets)
		if (use_gpu):
			_, preds = torch.max(out.cpu(),1)
		else:
			_, preds = torch.max(out, 1)
		running_loss = running_loss + float(loss.item())
		if (use_gpu):
			running_acc += torch.sum(preds==targets.data.cpu())
		else:
			running_acc += torch.sum(preds==targets.data)
		t.set_description("Epoch %d"%epoch)
		t.set_postfix(l=loss.item(), loss=running_loss/l, acc=running_acc/l2)
#		print("\repoch %d: %d  loss: %.3f"%(epoch+1, i+1, running_loss/(i+1)), end="")
		if (print_info): print(i,"/",l,'    loss:',running_loss/l,"accuracy:",running_acc/l2)
	ans.append(running_acc/l2)
#	print("\n")
	if (print_info): print("\n")
print(ans)
