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
from train import yaml_loader
import json 

def get_args():
    parser = argparse.ArgumentParser(description="Generate captions for an image using a trained model")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image or directory")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to the vocabulary file.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file.")
    parser.add_argument("--decoding_strategy", type=str, default="greedy", choices=["greedy", "beam_search", "min_p", "top_k"], help="Decoding strategy to use for caption generation.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--p", type=float, default = 0.95, required=False, help="p value for min_p")
    parser.add_argument("--k", type=int, default = 1000, required=False, help="k value for top_k")
    parser.add_argument("--beam", type=int, default=5, required=False, help="beam size for beam search")
    parser.add_argument("--max_len", type=int, default = 30, required=False, help="max_length for generated caption")
    parser.add_argument("--num_images", type=int, required=False, help="max number of images to generated caption")
    parser.add_argument('--save_img', action='store_true', help='Whether to save the image or not')
    return parser.parse_args()

def load_image(image_path, save_img = False):
    mean = (0.444, 0.421, 0.385)
    std = (0.285, 0.277, 0.286)
    img_transforms = v2.Compose([
        v2.ToImage(),
        v2.Resize(config['data']['img_size'], antialias = True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean = mean, std=std)
    ])
    raw_image = io.read_image(image_path)
    transformed_tensor = img_transforms(raw_image) 
    if save_img:
        torchvision.utils.save_image(transformed_tensor, 'image.png') 
    return transformed_tensor.unsqueeze(0)  # Add batch dimension

def generate_caption(model_path, image_path, vocab_path, decoding_strategy, save_img = False, **kwargs):
    vocab = pickle.load(open(args.vocab_path, 'rb'))
    model  = CaptionModel(**{**config['model_params'], "vocab_size":len(vocab.itos.keys())}).to(device) 
    state_dict = torch.load(args.model_path, map_location=device)
    print(model.load_state_dict(state_dict))
    model.eval()
    decoder = Decoding(model, vocab)
    strategy_function = {
        "greedy": decoder.greedy, 
        "beam_search": decoder.beam_search,
        "min_p": decoder.min_p,
        "top_k": decoder.top_k
    }

    all_image_paths = []

    if os.path.isdir(image_path):
        all_image_paths = list(map(lambda x: os.path.join(image_path, x), os.listdir(image_path)))
    else:
        all_image_paths = [image_path]

    if args.num_images:
        all_image_paths = all_image_paths[:args.num_images]

    caption_list = {}

    for idx, img_path in enumerate(all_image_paths):
        image = load_image(img_path, save_img = save_img)
        image = image.to(device)
        caption = decoder.get_caption(strategy_function[decoding_strategy](image, **kwargs))
        caption_list[img_path] = caption
        progress(idx+1, len(all_image_paths))
    return caption_list

if __name__ == "__main__":

    args = get_args()
    config = yaml_loader(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params_for_decoding = {
        "greedy": {"max_len": args.max_len},
        "beam_search": {"max_len": args.max_len, "beam_width": args.beam},
        "min_p": {"max_len": args.max_len, "p": args.p},
        "top_k": {"max_len": args.max_len, "k": args.k}
    }
    cur_decoding_params = params_for_decoding[args.decoding_strategy]
    generated_caption_list = generate_caption(args.model_path, args.image_path, args.vocab_path, args.decoding_strategy, args.save_img, **cur_decoding_params)
    for i,j in generated_caption_list.items():
        print(f"{i}: {j}")
    
    if len(generated_caption_list) > 1:
        os.makedirs("generated_captions", exist_ok = True)
        file_path = os.path.join("generated_captions", ".".join(args.model_path.split("/")[-1].split(".")[:-1]) + "_" + args.image_path.split("/")[-1] + f"_{args.num_images if args.num_images else ''}" + '.json')
        print(f"saving to: {file_path}")
        with open(file_path, "w") as f:
            json.dump(generated_caption_list, f, indent = 4)
