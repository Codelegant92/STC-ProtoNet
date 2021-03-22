import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import os
import json
import random
import re
import csv

cwd = os.getcwd()
data_path = join(cwd, 'speech_commands')
savedir = './'
dataset_list = ['base','base_unk','base_sil', 'val','val_unk','val_sil', 'novel','novel_unk', 'novel_sil']

cl = -1
folderlist = []

datasetmap = {'base':'2020_train_commands','val':'2020_val_commands','novel':'2020_test_commands',\
'base_unk':'2020_unk_train','val_unk':'2020_unk_val','novel_unk':'2020_unk_test',\
'base_sil':'2020_silence_train','val_sil':'2020_silence_val','novel_sil':'2020_silence_test'}
filelists = {'base':{},'val':{},'novel':{},'base_unk':{},'val_unk':{},'novel_unk':{},'base_sil':{},'val_sil':{},'novel_sil':{}}
filelists_flat = {'base':[],'val':[],'novel':[],'base_unk':[],'val_unk':[],'novel_unk':[],'base_sil':[],'val_sil':[],'novel_sil':[]}
labellists_flat = {'base':[],'val':[],'novel':[],'base_unk':[],'val_unk':[],'novel_unk':[],'base_sil':[],'val_sil':[],'novel_sil':[]}


for dataset in dataset_list:
	with open(datasetmap[dataset] + '.csv') as csvfile:
		csvreader = csv.reader(csvfile, delimiter=',')
		#next(csvreader, None)  # skip (filename, label)
		for i, row in enumerate(csvreader):
			filename = row[0]
			label = row[1]
			if not label in filelists[dataset]:
				folderlist.append(label)
				filelists[dataset][label] = []
			filelists[dataset][label].append(join(data_path,filename))

	for key, filelist in filelists[dataset].items():
		cl += 1
		random.shuffle(filelist)
		filelists_flat[dataset] += filelist
		labellists_flat[dataset] += np.repeat(cl, len(filelist)).tolist()

for dataset in dataset_list:
	fo = open(savedir+dataset+'.json', 'w')
	fo.write('{"label_names": [')
	fo.writelines(['"%s",' % item for item in folderlist])
	fo.seek(0, os.SEEK_END)
	fo.seek(fo.tell()-1, os.SEEK_SET)
	fo.write('],')

	fo.write('"image_names": [')
	fo.writelines(['"%s",' % item for item in filelists_flat[dataset]])
	fo.seek(0, os.SEEK_END)
	fo.seek(fo.tell()-1, os.SEEK_SET)
	fo.write('],')

	fo.write('"image_labels": [')
	fo.writelines(['%d,' % item for item in labellists_flat[dataset]])
	fo.seek(0, os.SEEK_END)
	fo.seek(fo.tell()-1, os.SEEK_SET)
	fo.write(']}')

	fo.close()
	print('%s -OK' % dataset)
		