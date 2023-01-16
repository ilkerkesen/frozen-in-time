import argparse

import pandas as pd
import torch
import transformers
from sacred import Experiment
from tqdm import tqdm
import glob
import data_loader.data_loader as module_data
import model.metric as module_metric
import model.model as module_arch
from model.model import compute_similarity
from parse_config import ConfigParser
from trainer.trainer import verbose
from utils.util import state_dict_data_parallel_fix
import numpy as np
import os
import copy

ex = Experiment('test')


@ex.main
def run():
    # setup data_loader instances
    config._config['data_loader']['args']['split'] = args.split
    config._config['data_loader']['args']['tsfm_split'] = 'test'  # set transform to test split to remove augmentations
    config._config['data_loader']['args']['shuffle'] = False
    config._config['data_loader']['args']['batch_size'] = args.batch_size
    config._config['data_loader']['args']['sliding_window_stride'] = args.sliding_window_stride
    config._config['data_loader']['args']['metadata_filename'] = args.metadata_filename
    config._config['data_loader']['args']['quva_dir'] = args.quva_dir
    config._config['data_loader']['args']['something_something_dir'] = args.something_something_dir
    # config._config['data_loader']['args']['video_params']['num_frames'] = 120

    data_loader = config.initialize('data_loader', module_data)
    n_samples = len(data_loader.dataset)

    text_model_name = config['arch']['args']['text_params']['model']
    if "openai/clip" in text_model_name:
        tokenizer_builder = transformers.CLIPTokenizer
    else:
        tokenizer_builder = transformers.AutoTokenizer
    tokenizer = tokenizer_builder.from_pretrained(
        text_model_name,
        model_max_length=config['arch']['args']['text_params'].get('max_length', 1e6),
        TOKENIZERS_PARALLELISM=False)

    # build model architecture
    model = config.initialize('arch', module_arch)

    # get function handles of loss and metrics
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    # logger.info('Loading checkpoint: {} ...'.format(config.resume))

    if config.resume is not None:
        checkpoint = torch.load(config.resume)
        state_dict = checkpoint['state_dict']
        new_state_dict = state_dict_data_parallel_fix(state_dict, model.state_dict())
        model.load_state_dict(new_state_dict, strict=True)
    else:
        print('Using random weights')

    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    ctr = 0
    save_part = None
    if args.save_feats:
        part_seq = [int(x.split('_')[-1].split('.')[0]) for x in
                    glob.glob(os.path.join(args.save_feats, "ids_test_*.csv"))]
        if len(part_seq) > 0:
            save_part = max() + 1
        else:
            save_part = 0
        print(F"##### WARNING SAVE_PART STARTING AT {save_part}, MAKE SURE THIS IS THE NEWEST")

    print(len(data_loader))
    num_correct, num_examples = 0, 0    
    with torch.no_grad():
        for i, data_og in tqdm(tqdm(enumerate(data_loader))):
            # leave this for now since not doing anything on the gpu
            data = copy.deepcopy(data_og)
            del data_og
            text_inputs = tokenizer(
                data['text'],
                return_tensors='pt',
                padding=True,
                truncation=True,
            ).to('cuda:0')
            data['video'] = data['video'].to(device)
            inputs = {
                'video': data['video'],
                'text': text_inputs,
            }
            text_embeds, vid_embeds = model(inputs)
            output, _ = compute_similarity(text_embeds, vid_embeds)
            import ipdb; ipdb.set_trace()
            batch_size, offset = vid_embeds.shape[0], 0
            for i in range(batch_size):
                num_text = data['num_text'][i]
                this = output[offset:offset+num_text, i]
                if this.argmax().item() == 0:
                    num_correct += 1
                num_examples += 1
                offset += num_text
    acc = round(100 * num_correct / num_examples, 2)
    print(f'accuracy={acc}%')


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-s', '--sliding_window_stride', default=-1, type=int,
                      help='test time temporal augmentation, repeat samples with different start times.')
    args.add_argument('--save_feats', default=None,
                      help='path to store text & video feats, this is for saving embeddings if you want to do offline retrieval.')
    args.add_argument('--save_type', default='both', choices=['both', 'text', 'video'],
                      help='Whether to save video, text or both feats. If running on inference videos, text is just a placeholder')
    args.add_argument('--vis_token_similarity', action='store_true')
    args.add_argument('--split', default='test', choices=['train', 'val', 'test'],
                      help='split to evaluate on.')
    args.add_argument('--batch_size', default=16, type=int,
                      help='size of batch')
    args.add_argument('--metadata_filename', default='relations.json',
                      help='annotations file name (e.g. relations.json)')
    args.add_argument('--quva_dir', default=None,
                      help='full path to the QUVA dataset root dir.')
    args.add_argument('--something_something_dir', default=None,
                      help='full path to the something something dataset (v2) video dir.')
    config = ConfigParser(args, test=True)
    # hack to get sliding into config
    args = args.parse_args()
    config._config['sliding_window_stride'] = args.sliding_window_stride
    ex.add_config(config.config)

    ex.run()
