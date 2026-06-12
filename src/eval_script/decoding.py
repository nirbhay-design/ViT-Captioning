import torch
import torch.nn.functional as F
class Decoding():
    def __init__(self, model, vocab):
        self.model = model
        self.vocab = vocab
        self.sos_token = self.vocab.stoi['<SOS>']

    def get_caption(self, y_out):
        captions = []
        for i in range(y_out.shape[0]):
            caption = self.vocab.decode(y_out[i].tolist())
            captions.append(caption)
        return captions

    def greedy(self, x, max_len=30):  
        embedding, pos_encoding = self.model.model.get_embedding(x)
        device = embedding.device
        batch_size = embedding.shape[0]
        y_out = torch.ones(batch_size, 1, device=device).type(torch.int64) # <SOS> token vector
        for _ in range(max_len):
            decoder_output = self.model.model.get_decoding(embedding, pos_encoding, y_out)
            _, predicted_words = decoder_output[:,-1:,:].max(dim=-1) # greedy 
            y_out = torch.cat([y_out, predicted_words], dim=1)
        return y_out

    def beam_search(self, x, max_len=30, beam_width=5):
        all_probs = {}
        #keep top k logits for first step, and then keep top k sequences
        embedding, pos_encoding = self.model.model.get_embedding(x)
        device = embedding.device
        batch_size = embedding.shape[0]
        y_out = torch.ones(batch_size, 1, device=device).type(torch.int64) # <SOS> token vector
        # for _ in range(max_len):
        decoder_output = self.model.model.get_decoding(embedding, pos_encoding, y_out)
        values, indicies = torch.topk(F.softmax(decoder_output[:,-1:,:], dim = -1), beam_width) # topk probs
        beam = indicies # (B, 1, k)
        probs = values
        while y_out.shape[1] < max_len:
            for i in range(beam_width):
                curr_inp = torch.cat([y_out, beam[:,:,i]], dim=1) # (B, seq_len+1)
                decoder_output = self.model.model.get_decoding(embedding, pos_encoding, curr_inp)
                values, indicies = torch.topk(F.softmax(decoder_output[:,-1:,:], dim = -1), beam_width) # topk probs
                curr_probs = probs[0,0,i]*values

                all_probs[i+1] = {}
                all_probs[i+1][beam[0,0,i].item()] = {curr_probs[0][0][j]: indicies[0][0][j].item() for j in range(beam_width)}

            print(all_probs)
            exit(0)
            
        
        return y_out

    def min_p(self, x, max_len=30, p = 0.7):
        embedding, pos_encoding = self.model.model.get_embedding(x)
        device = embedding.device
        batch_size = embedding.shape[0]
        y_out = torch.ones(batch_size, 1, device=device).type(torch.int64) # <SOS> token vector
        for _ in range(max_len):
            decoder_output = self.model.model.get_decoding(embedding, pos_encoding, y_out)
            probs = F.softmax(decoder_output[:,-1:,:], dim = -1) # topk probs
            threshold = p*probs.max()
            mask = probs >= threshold
            y = torch.ones_like(probs)*torch.tensor(-1000000.0)
            alive = F.softmax(torch.where(probs[0,0,:]>=threshold.item(), probs[0,0,:], y[0,0,:]), dim=-1)
            sampled_index = alive.multinomial(num_samples=1, replacement=True)
            # print(sampled_index.unsqueeze(0).shape)
            y_out = torch.cat([y_out, sampled_index.unsqueeze(0)], dim=1)
        return y_out
        

    def top_k(self, x, max_len=30, k=5):
        embedding, pos_encoding = self.model.model.get_embedding(x)
        device = embedding.device
        batch_size = embedding.shape[0]
        y_out = torch.ones(batch_size, 1, device=device).type(torch.int64)
        for _ in range(max_len):
            decoder_output = self.model.model.get_decoding(embedding, pos_encoding, y_out)
            _, indicies = decoder_output[:,-1:,:].topk(k, dim = -1) # topk probs
            sampled_index = indicies.flatten()[torch.randint(0, indicies.numel(), (1,))]
            # print(sampled_index.unsqueeze(0).shape)
            y_out = torch.cat([y_out, sampled_index.unsqueeze(0)], dim=1)

        return y_out