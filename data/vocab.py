import string
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

class Vocabulary():
    def __init__(self, feq, tok):
        self.feq = feq
        self.itos = {
            0:"<PAD>",
            1:"<SOS>",
            2:"<EOS>",
            3:"<UNK>"
        }
        self.stoi = {j:i for i,j in self.itos.items()}
        self.tok = tok

    def tokenizer(self, text):
        return [tok for tok in self.tok(text)]

    def build_voc(self, text_list):
        idx = len(self.itos)
        curfeq = {}
        for text in text_list:
            for word in self.tokenizer(text):
                if word not in curfeq:
                    curfeq[word] = 1
                else:
                    curfeq[word] += 1
                if curfeq[word] == self.feq:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def encode(self, text):
        tokenize_text = self.tokenizer(text)

        numeric_val = [self.stoi['<SOS>']]
        numeric_val += [self.stoi[token] if token in self.stoi else self.stoi['<UNK>'] for token in tokenize_text]
        numeric_val += [self.stoi['<EOS>']]

        return numeric_val

    def __len__(self):
        return len(self.itos)

    def decode(self, numeric_val):
        output_text = ""
        for i in numeric_val:
            if i == self.stoi['<EOS>']:
                output_text += '<EOS>'
                break
            output_text += self.itos[i] + " "
        
        return output_text

class Vocab:
    def __init__(self, t):
        self.t = t
        self.itos = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
    def clean_caption(self, caption):
        caption = caption.lower().translate(str.maketrans("", "", string.punctuation))
        return caption
    def flatten(self, xss):
        return [self.clean_caption(x) for xs in xss for x in xs]
    def get_counts(self,x):
        counts = Counter()
        flattened = self.flatten(x)
        for text in flattened:  
            counts.update(text.split())
        counts = {x: count for x, count in counts.items() if count >= self.t}
        self.itos.update({i+4: k for i, k in enumerate(counts.keys())})
        self.stoi = {v: k for k, v in self.itos.items()}
    def encode(self, text):
        tokenize_text = self.clean_caption(text).split()
        # print(tokenize_text)

        numeric_val = [self.stoi['<SOS>']]
        numeric_val += [self.stoi[token] if token in self.stoi else self.stoi['<UNK>'] for token in tokenize_text]
        numeric_val += [self.stoi['<EOS>']]

        return numeric_val

    def __len__(self):
        return len(self.itos)

    def decode(self, numeric_val):
        output_text = ""
        for i in numeric_val:
            if i == self.stoi['<EOS>']:
                output_text += '<EOS>'
                break
            output_text += self.itos[i] + " "
        
        return output_text
