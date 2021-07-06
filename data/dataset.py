# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import json
import numpy as np
import torchvision.transforms as transforms
import os
import librosa
import random
from skimage.transform import resize
from glob import glob

identity = lambda x:x

class ReadAudio:
    def __init__(self):
        self.sr = 16000
        self.background_noises = [librosa.load(x, sr=self.sr)[0] for x in glob("./filelists/speech_commands/_background_noise_/*.wav")]

    def get_one_noise(self):
        #print("length of background noise: %d" % len(self.background_noises))
        selected_noise = self.background_noises[random.randint(0, len(self.background_noises)-1)]
        start_idx = random.randint(0, len(selected_noise)-1-self.sr)
        return selected_noise[start_idx:(start_idx+self.sr)]

    def get_mix_noises(self, num_noise=1, max_ratio=0.1):
        result = np.zeros(self.sr)
        for _ in range(num_noise):
            result += random.random()*max_ratio*self.get_one_noise()
        return result / num_noise if num_noise > 0 else result

    def get_silent_wav(self, num_noise=1, max_ratio=0.1):
        return self.get_mix_noises(num_noise=num_noise, max_ratio=max_ratio)

    def get_one_word_wav(self, path, speed_rate=None):
        wav = librosa.load(path, sr=self.sr)[0]
        if speed_rate:
            wav = librosa.effects.time_stretch(wav, speed_rate)
        if len(wav) < self.sr:
            wav = np.pad(wav, (0, self.sr - len(wav)), 'constant')
        return wav[:self.sr]

    def preprocess_mfcc(self, wave):
        spectrogram = librosa.feature.melspectrogram(wave, sr=self.sr, n_mels=40, hop_length=160, n_fft=480, fmin=20, fmax=4000)
        #idx = [spectrogram > 0]
        #spectrogram[tuple(idx)] = np.log(spectrogram[tuple(idx)])
        #dct_filters = librosa.filters.dct(n_filters=40, n_input=40)
        #mfcc = [np.matmul(dct_filters, x) for x in np.split(spectrogram, spectrogram.shape[1], axis=1)]
        #mfcc = np.hstack(mfcc)
        #mfcc = mfcc.astype(np.float32)
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(spectrogram))
        return mfcc

class SimpleDataset:
    def __init__(self, data_file, fixed_data_file_list, resize, transform, target_transform=identity):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        self.meta_fixed_classes = []
        for fixed_data_file in fixed_data_file_list:
            with open(fixed_data_file, 'r') as f:
                self.meta_fixed_classes.append(json.load(f))
        self.transform = transform
        self.target_transform = target_transform
        self.ReadAudio = ReadAudio()
        self.resize = resize

        self.image_path_names = self.meta['image_names']
        self.image_label_names = self.meta['image_labels']

        for meta_fixed_class in self.meta_fixed_classes:
            self.image_path_names.extend(meta_fixed_class['image_names'])
            self.image_label_names.extend(meta_fixed_class['image_labels'])


    def __getitem__(self,i):

        image_path = os.path.join(self.image_path_names[i])
        if 'silence' in image_path:
            img = torch.from_numpy(resize(self.ReadAudio.preprocess_mfcc(self.ReadAudio.get_silent_wav(num_noise=random.choice([0,1,2,3]),\
                max_ratio=random.choice([x/10. for x in range(20)]))), (self.resize, self.resize), preserve_range=True)).float().view(1, self.resize, self.resize)
        else:
            img = torch.from_numpy(resize(self.ReadAudio.preprocess_mfcc(self.ReadAudio.get_one_word_wav(image_path)), (self.resize, self.resize), preserve_range=True)).float().view(1, self.resize, self.resize)
        
        target = self.target_transform(self.image_label_names[i])
        return img, target

    def __len__(self):
        return len(self.image_path_names)


class SetDataset:
    def __init__(self, data_file, fixed_data_file_list, resize, batch_size, transform):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        self.meta_fixed_classes = []
        for fixed_data_file in fixed_data_file_list:
            with open(fixed_data_file, 'r') as f:
                self.meta_fixed_classes.append(json.load(f))
 
        self.cl_all_list = np.unique(self.meta['image_labels']).tolist()
        for meta_fixed_class in self.meta_fixed_classes:
            self.cl_all_list.extend(np.unique(meta_fixed_class['image_labels']).tolist())

        self.sub_meta_all = {}
        for cl in self.cl_all_list:
            self.sub_meta_all[cl] = []

        for x,y in zip(self.meta['image_names'],self.meta['image_labels']):
            self.sub_meta_all[y].append(x)

        for meta_fixed_class in self.meta_fixed_classes:
            for x,y in zip(meta_fixed_class['image_names'],meta_fixed_class['image_labels']):
                self.sub_meta_all[y].append(x)

        self.sub_dataloader = [] 
        sub_data_loader_params = dict(batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 0, #use main thread only or may receive multiple batches
                                  pin_memory = False)        
        for cl in self.cl_all_list:
            sub_dataset = SubDataset(self.sub_meta_all[cl], cl, resize, transform=transform)
            self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params))

    def __getitem__(self,i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.cl_all_list)

class SubDataset:
    def __init__(self, sub_meta_all, cl, resize, transform=transforms.ToTensor(), target_transform=identity):
        self.sub_meta_all = sub_meta_all
        self.cl = cl 
        self.transform = transform
        self.target_transform = target_transform
        self.resize = resize
        self.ReadAudio = ReadAudio()

    def __getitem__(self,i):
        image_path = os.path.join( self.sub_meta_all[i])
        if 'silence' in image_path:
            img = torch.from_numpy(resize(self.ReadAudio.preprocess_mfcc(self.ReadAudio.get_silent_wav(num_noise=random.choice([0,1,2,3]),\
                max_ratio=random.choice([x/10. for x in range(20)]))), (self.resize, self.resize), preserve_range=True)).float().view(1, self.resize, self.resize)
        else:
            img = torch.from_numpy(resize(self.ReadAudio.preprocess_mfcc(self.ReadAudio.get_one_word_wav(image_path)), (self.resize, self.resize), preserve_range=True)).float().view(1, self.resize, self.resize)
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.sub_meta_all)

class EpisodicBatchSampler(object):
    def __init__(self, n_classes_all, n_classes_fixed, n_way, max_n_way, min_n_way, n_episodes):
        self.n_classes_all = n_classes_all
        self.n_classes_fixed = n_classes_fixed
        self.n_classes_dynamic = self.n_classes_all - self.n_classes_fixed
        self.n_way = n_way
        self.max_n_way = max_n_way
        self.min_n_way = min_n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            #yield torch.randperm(self.n_classes_all)[:self.n_way]
            if self.n_way == -1:
                selected_n_way = np.random.randint(self.min_n_way, self.max_n_way+1)
            else:
                selected_n_way = self.n_way
            yield torch.cat((torch.randperm(self.n_classes_dynamic)[:(selected_n_way-self.n_classes_fixed)], torch.arange(self.n_classes_all)[-self.n_classes_fixed:]))
