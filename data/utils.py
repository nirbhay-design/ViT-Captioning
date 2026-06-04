import glob
import json
train_files = glob.glob('../dataset/train2014/*.jpg')
val_files = glob.glob('../dataset/val2014/*.jpg')
captions = json.load(open('../dataset/annotations/captions_train2014.json'))
print('Number of training files: {}'.format(len(train_files)))
print('Number of validation files: {}'.format(len(val_files)))
print(train_files[:5])
print(captions['info']['url'][:2])
print(captions['annotations'][:2])
print(captions['images'][:2])