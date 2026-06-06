import re 
import torch 
import string

class CustomCollate():
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):

        image = []
        text_eng = []

        for bt in batch:
            image.append(bt[0].unsqueeze(0))
            text_eng.append(bt[1])

        padded_text_eng = torch.nn.utils.rnn.pad_sequence(text_eng, batch_first = True, padding_value = self.pad_idx)
        image = torch.cat(image, dim=0)

        return image, padded_text_eng

class Preprocessor():
    
    def preprocess(self, text):
        text = text.lower()#converting string to lowercase
        res1 = re.sub(r'((http|https)://|www.).+?(\s|$)',' ',text)#removing links
        res2 = re.sub(f'[{string.punctuation}]+',' ',res1)#removing non english and special characters
        res3 = re.sub(r'[^a-z0-9A-Z\s]+',' ',res2)#removing anyother that is not consider in above
        res4 = re.sub(r'(\n)+',' ',res3)#removing all new line characters
        res = re.sub(r'\s{2,}',' ',res4)#remove all the one or more consecutive occurance of sapce
        res = res.strip()
        return res