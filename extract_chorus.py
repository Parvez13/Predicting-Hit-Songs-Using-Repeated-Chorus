from pychorus import find_and_output_chorus



def extract_song_chorus(path, main):
	# songname = path.split('/',2)[0].split('.')[0]
	Newpath =  main + '/' + "song_to_predict"+'.wav'

	chorus = find_and_output_chorus(path, Newpath, 15)
	if chorus == None:
		return None

	else:
		return Newpath

