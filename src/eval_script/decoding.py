class Decoding():
    def __init__(self, model, vocab):
        self.model = model
        self.vocab = vocab
        self.sos_token = self.vocab.stoi['<SOS>']

    def greedy(self, x, max_len=30):  
        embedding, pos_encoding = self.model.model.get_embedding(x)
        device = embedding.device
        batch_size = embedding.shape[0]
        y_out = torch.ones(batch_size, 1, device=device).type(torch.int64) # <SOS> token vector

        


    def beam_search(self, x, max_len=30, beam_width=5):
        pass 

    def min_p(self, x, max_len=30):
        pass

    def top_k(self, x, max_len=30, k=5):
        pass