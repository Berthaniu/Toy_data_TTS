import numpy as np 
import pdb
import os

class Generator():

	def __init__(
		self, n_vocab, freq):
		super(Generator,self).__init__()

		self.freq = freq
		self.n_vocab = n_vocab

	def gaussian_rmv(self,mean,std,size):
		'''
			Generate gaussian random int with given mean and var.
		'''

		dur = np.random.normal(mean,std,size)

		return dur
	
	def multivariate_normal(self,mean,std,dur_size):
		# Generate multi variate mean 1xN
		mean_n = self.gaussian_rmv(mean,5,dur_size)

		# Generate NxN variance matrix
		var_n = self.gaussian_rmv(std,1,dur_size)
		var_n =np.abs(var_n)

		# Generate multivariate normal  [freq,dur_size]
		sample = self.gaussian_rmv(mean_n,var_n,(self.freq,dur_size))
		return sample


	def text_generator(self,text_len):

		# Generate text_len random variable represents duration_len for each text.
		text_rep = np.random.randint(1,self.n_vocab, size=text_len) 
		text_dur = np.random.randint(2,10, size=text_len) 
		
		# Each text, generate spectrogram sample with unique distribution.
		for (t,dur_size) in enumerate(text_dur):
			mean = np.random.randint(999)*np.random.randn()
			var = np.random.randint(25)
			sample = self.multivariate_normal(mean,np.sqrt(var),dur_size)
			# Add gaussian noises
			eps = self.gaussian_rmv(0,1,(self.freq,dur_size))
			mel_ = sample+eps
			if t ==0:
				mel = mel_
			else:
				mel = np.append(mel,mel_,1)
		# scale mel into [-1,1]	
		mel= (mel - mel.min())/(mel.max()-mel.min())*2 - 1
		mel_len = mel.shape[-1]	
		return text_rep,text_dur,mel,mel_len

	def save_data(self,data,path):
		with open(path,'wb') as f:
			np.save(f,data)
			f.close()




if __name__ == "__main__":

	n_vocab = 362
	freq = 80
	min_len = 20
	max_len = 150
	n = 10
	max_mel_len = 1500
	
	Generator = Generator(n_vocab,freq)

	# Randomly generate n text lengths
	text_len = np.random.randint(min_len,max_len, n)
	
	mel_save = np.zeros((n,freq,max_mel_len))
	mel_len_save = np.zeros(n)
	text_dur_save = np.zeros((n,max_len))
	text_rep_save = np.zeros((n,max_len))
	text_len_save = np.zeros(n)
	for (idx,l) in enumerate(text_len):
		# Generate one sample
		text_rep,text_dur,mel,mel_len = Generator.text_generator(l)
		# Store sample
		mel_save[idx,:,:mel_len] = mel
		mel_len_save[idx] = mel_len

		text_dur_save[idx,:l] = text_dur
		text_rep_save[idx,:l] = text_rep
		text_len_save[idx] = l

	# Save data
	if not os.path.exists('./data'):
		os.mkdir('./data')

	Generator.save_data(mel_save,'./data/mel.npy')
	Generator.save_data(mel_len_save,'./data/mel_len.npy')
	Generator.save_data(text_dur_save,'./data/text_dur.npy')	
	Generator.save_data(text_len_save,'./data/text_len.npy')
	Generator.save_data(text_rep_save,'./data/text.npy')


