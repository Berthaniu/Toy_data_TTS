import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pdb
import os

class toydata(Dataset):
	def __init__(self, root_dir):
		self.root_dir = root_dir
		self.size =  len(np.load(os.path.join(self.root_dir,'mel_len.npy')))


	def read_npy(self):
		mel = np.load(os.path.join(self.root_dir,'mel.npy'))
		mel_len = np.load(os.path.join(self.root_dir,'mel_len.npy'))
		text_rep = np.load(os.path.join(self.root_dir,'text.npy'))
		text_dur = np.load(os.path.join(self.root_dir,'text_dur.npy'))
		text_len = np.load(os.path.join(self.root_dir,'text_len.npy'))
		return mel,mel_len,text_rep,text_dur,text_len

	def __len__(self):
		return self.size

	def __getitem__(self,idx):
		# Read data
		mel,mel_len,text_rep,text_dur,text_len = self.read_npy()
		# Get sample
		mel_ = mel[idx,:,:]
		mel_len_ = mel_len[idx]
		text_rep_ = text_rep[idx,:]
		text_dur_ = text_dur[idx,:]
		text_len_ = text_len[idx]
		merge = {'mel':mel_,'mel_length':mel_len_,'text':text_rep_,
		 'duration': text_dur_,'text_length':text_len_}
		return merge

if __name__ == "__main__":
	'''
		Test the dataloader
	'''
dataset = toydata('./data')

for i in range(len(dataset)):
	sample = dataset[i]
	print(sample.keys())