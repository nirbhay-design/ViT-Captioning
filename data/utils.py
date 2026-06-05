import glob
import json
import numpy as np
import torchvision.io as io
## extract image id and map all annotations of an id to the image
import torch
class cocoDataset:
    def __init__(self, path, train=True):
        self.train = train
        self.path = path
        if self.train:
            self.files = glob.glob(self.path + 'train2014/*.jpg')
            self.captions = json.load(open(self.path + 'annotations/captions_train2014.json'))
        else:
            self.files = glob.glob(self.path + 'val2014/*.jpg')
            self.captions = json.load(open(self.path + 'annotations/captions_val2014.json'))
        # self.train_files = glob.glob(self.path + 'train2014/*.jpg')
        # self.val_files = glob.glob(self.path + 'val2014/*.jpg')
        
        self.id_to_captions = {}
        for a in self.captions['annotations']:
            image_id = a['image_id']
            caption = a['caption']
            if image_id not in self.id_to_captions:
                self.id_to_captions[image_id] = []
            self.id_to_captions[image_id].append(caption)
# to do : extract image id from file name and map to the captions   
    def __getitem__(self,idx):
        file = self.files[idx]
        image = io.read_image(file)
        image_id = int(file.split('.jpg')[0].split('_')[-1])
        captions = self.id_to_captions[image_id]
        len_captions = len(captions)
        # return image for each caption
        # do i still need a collate fn?
        # i have a feeling it can work without it but i am not certain
        # wait does the tokenization happen here? prolly not 
        # ok ask nirbhay
        return torch.stack([image] * len_captions), captions
    
    def __len__(self):
        return len(self.files)


    

if __name__ == '__main__':
    dataset = cocoDataset('../dataset/')
    print('Number of training files: {}'.format(len(dataset.files)))
    print(dataset.__getitem__(00))
    




# print('Number of training files: {}'.format(len(train_files)))
# print('Number of validation files: {}'.format(len(val_files)))
# print(train_files[:5])
# print(captions['info']['url'][:2])
# print(captions['annotations'][:2])
# print(captions['images'][:2])