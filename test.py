# TODO - image path, vocab path, model path
# TODO - use Decoding class
# TODO - image transformations from train.py, model loading from train.py, config from train.py 
# TODO - based on command line argument - flip between greedy, beam search, min_p, top_k

import torchtext; torchtext.disable_torchtext_deprecation_warning()
from data.vocab import Vocab
import torch, os
import pickle
import torchvision 
from utils import progress
from torchvision.transforms import v2
import torchvision.io as io
from torchtext.data import get_tokenizer
import argparse
from src.caption_model import *
from src.eval_script.decoding import Decoding
from data.vocab import Vocabulary
from utils import yaml_loader
from data import dataloaders
import json 
import time 

def get_args():
    parser = argparse.ArgumentParser(description="Generate captions for an image using a trained model")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to the vocabulary file.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file.")
    parser.add_argument("--decoding_strategy", type=str, nargs='+', default=["greedy", "min_p", "top_k", "top_p"], help="Decoding strategy to use for caption generation.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--min_p", type=float, default = 0.05, required=False, help="p value for min_p")
    parser.add_argument("--top_p", type=float, default = 0.95, required=False, help="p value for top_p")
    parser.add_argument("--k", type=int, default = 50, required=False, help="k value for top_k")
    parser.add_argument("--beam", type=int, default=5, required=False, help="beam size for beam search")
    parser.add_argument("--max_len", type=int, default = 30, required=False, help="max_length for generated caption")
    parser.add_argument("--temp", type=float, default = 0.7, required=False, help="temperature value for min_p or top_p")
    parser.add_argument('--bs', type=int, default=64, help='Batch size')
    parser.add_argument('--nw', type=int, default=4, help='Number of workers')
    parser.add_argument('--pf', type=int, default=2, help='Prefetch factor')
    return parser.parse_args()

def generate_caption(model_path, test_loader, vocab, config, decoding_strategy, params_for_decoding):
    model  = CaptionModel(**{**config['model_params'], "vocab_size":len(vocab.itos.keys())}).to(device) 
    state_dict = torch.load(model_path, map_location=device)
    print(model.load_state_dict(state_dict))
    model.eval()
    decoder = Decoding(model, vocab)
    strategy_function = {
        "greedy": decoder.greedy, 
        "beam_search": decoder.beam_search,
        "min_p": decoder.min_p,
        "top_k": decoder.top_k,
        "top_p": decoder.top_p
    }

    caption_list = {}

    for idx, (image, caption) in enumerate(test_loader):
        image = image.to(device)
        emb, pos_enc = model.model.get_embedding(image)
        caption_decoding_strategy = {}
        caption_decoding_strategy["original"] = decoder.get_caption(caption)
        for strategy in decoding_strategy:
            cur_caption = decoder.get_caption(strategy_function[strategy](emb=emb, pos_enc=pos_enc, **params_for_decoding[strategy]))
            caption_decoding_strategy[strategy] = cur_caption
        for i in range(image.shape[0]):
            caption_list[f"{idx}_{i}"] = {
                "original": caption_decoding_strategy["original"][i],
                **{stg: caption_decoding_strategy[stg][i] for stg in decoding_strategy}
            }
        # progress(idx+1, len(test_loader))
    return caption_list

if __name__ == "__main__":

    args = get_args()
    config = yaml_loader(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config["data"]["num_workers"] = args.nw 
    config["data"]["prefetch_factor"] = args.pf
    config["data"]["batch_size"] = args.bs
    config["data"]["vocab_save_path"] = args.vocab_path

    dl = dataloaders["coco"](config["data"])
    test_loader = dl['test_loader']

    params_for_decoding = {
        "greedy": {"max_len": args.max_len},
        "beam_search": {"max_len": args.max_len, "beam_width": args.beam, "temp": args.temp},
        "min_p": {"max_len": args.max_len, "p": args.min_p, "temp": args.temp},
        "top_k": {"max_len": args.max_len, "k": args.k, "temp": args.temp},
        "top_p": {"max_len": args.max_len, "p": args.top_p, "temp": args.temp}
    }
    vocab = dl["vocab"]
    generated_caption_list = generate_caption(args.model_path, test_loader, vocab, config, args.decoding_strategy, params_for_decoding)

    file_path = os.path.join("generated_captions", ".".join(args.model_path.split("/")[-1].split(".")[:-1]) + "_" + ".".join(args.decoding_strategy) + '.json')
    print(f"saving to: {file_path}")
    with open(file_path, "w") as f:
        json.dump(generated_caption_list, f, indent = 4)    