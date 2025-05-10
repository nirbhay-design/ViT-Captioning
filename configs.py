class Args():
    def __init__(self, default_args=None):

        self.encode_names = {}

        if default_args:
            for key, value in default_args.items():
                self.__dict__[key] = value
                self.encode_names[f'--{key}'] = key

    def add_args(self, name, default_value):
        name_val = name[2:]
        self.encode_names[name] = name_val 
        self.__dict__[name_val] = default_value
    
    def build_args(self, sys_argv):
        sys_argv_useful = sys_argv[1:]

        i = 0
        while (i < len(sys_argv_useful)): 
            cur_val = sys_argv_useful[i]
            if cur_val in self.encode_names:
                arg_value = self.__dict__[self.encode_names[cur_val]]
                updated_arg_value = sys_argv_useful[i+1]                
                if isinstance(arg_value, bool):
                    updated_arg_value = eval(updated_arg_value)
                elif isinstance(arg_value, int):
                    updated_arg_value = int(updated_arg_value)
                elif isinstance(arg_value, float):
                    updated_arg_value = float(updated_arg_value)
                elif isinstance(arg_value, str):
                    updated_arg_value = str(updated_arg_value)
                
                self.__dict__[self.encode_names[cur_val]] = updated_arg_value
                i += 2
            else:
                i+=1

    def print_args(self):
        print("===> Configurations")
        for i in self.encode_names.values():
            print(f'{i} ==> {self.__dict__[i]}')

def build_args(sys_argv):

    default_args = {
        # data configs
        'img_size': [256, 340],
        'image_path': "/DATA/dataset/Flickr30k/Flickr30k/Images",
        'captions_path': "/DATA/dataset/Flickr30k/Flickr30k/captions.txt",
        'batch_size': 32,
        'pin_memory': True,
        'num_workers': 4,

        # detr configs
        'backbone_layers': ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4'],
        'encoder_layers': 6,
        'decoder_layers': 6,
        'encoder_heads': 8,
        'decoder_heads': 8,
        'embed_dim': 256,
        'dropout': 0.1,

        # training configs
        'lr': 0.0001,
    }

    arg = Args(default_args)

    arg.add_args('--gpu', 5)
    arg.add_args('--return_logs', False)
    arg.add_args('--vocabulary_size', 20000)
    arg.add_args('--save_path', 'saved_models/detr_img_caption_v1.pth')
    arg.add_args('--epochs', 100)

    arg.build_args(sys_argv)

    return arg


def build_args_test(sys_argv):

    default_args = {
        # data configs
        'img_size': [256, 340],
        'image_path': "/DATA/dataset/Flickr30k/Flickr30k/Images",
        'captions_path': "/DATA/dataset/Flickr30k/Flickr30k/captions.txt",
        'dataset':'Flickr30k',
        'batch_size': 32,
        'pin_memory': True,
        'num_workers': 4,

        # detr configs
        'backbone_layers': ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4'],
        'encoder_layers': 6,
        'decoder_layers': 6,
        'encoder_heads': 8,
        'decoder_heads': 8,
        'embed_dim': 256,
        'dropout': 0.1
    }

    arg = Args(default_args)

    arg.add_args('--gpu', 5)
    arg.add_args('--return_logs', False)
    arg.add_args('--vocabulary_size', 20000)
    arg.add_args('--vocab_path', 'vocab/vocab.pkl')
    arg.add_args('--model_path', 'saved_models/detr_img_caption_v1.pth')

    arg.build_args(sys_argv)

    return arg

def build_args_test_sample(sys_argv):

    default_args = {
        # data configs
        'img_size': [256, 340],

        # detr configs
        'backbone_layers': ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4'],
        'encoder_layers': 6,
        'decoder_layers': 6,
        'encoder_heads': 8,
        'decoder_heads': 8,
        'embed_dim': 256,
        'dropout': 0.1
    }

    arg = Args(default_args)

    arg.add_args('--gpu', 5)
    arg.add_args('--vocab_path', 'vocab/vocab.pkl')
    arg.add_args('--model_path', 'saved_models/detr_img_caption_v1.pth')
    arg.add_args('--image_path', "path_to_img")
    arg.add_args('--vocabulary_size', 20000)

    arg.build_args(sys_argv)

    return arg