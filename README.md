## Toy_data_TTS
A toy dataset for TTS-duration alignment 

## dataGenerator.py
Return: Aligned text_rep,text_dur,mel,mel_len 

max_mel_len:1200
max_text_len: 150

Save path:'./data'

Save type: '.npy' files

## dataloader.py
Read the generated toy dataset and return a dict as 
\{'mel':mel_,'mel_length':mel_len_,'text':text_rep_,
		 'duration': text_dur_,'text_length':text_len_\}
