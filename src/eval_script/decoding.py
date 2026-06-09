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
            y_out = torch.cat([y, predicted_words], dim=1)
        return y_out

    def beam_search(self, x, max_len=30, beam_width=5):
        pass 

    def min_p(self, x, max_len=30):
        pass

    def top_k(self, x, max_len=30, k=5):
        pass