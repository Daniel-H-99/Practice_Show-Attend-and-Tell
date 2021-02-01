import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision
import torchvision.datasets
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import TensorDataset, DataLoader 
from dataloader import get_k_fold_loader, preprocess
from dataset.field import Vocab
from utils import *
from model import SAT
from tqdm import tqdm
import logging
import time
from PIL import Image
    
def main(args):
    
    # 0. initial setting
    
    # set environmet
    cudnn.benchmark = True

    if not os.path.isdir(os.path.join(args.path, './ckpt')):
        os.mkdir(os.path.join(args.path,'./ckpt'))
    if not os.path.isdir(os.path.join(args.path,'./results')):
        os.mkdir(os.path.join(args.path,'./results'))    
    if not os.path.isdir(os.path.join(args.path, './ckpt', args.name)):
        os.mkdir(os.path.join(args.path, './ckpt', args.name))
    if not os.path.isdir(os.path.join(args.path, './results', args.name)):
        os.mkdir(os.path.join(args.path, './results', args.name))
    if not os.path.isdir(os.path.join(args.path, './results', args.name, "log")):
        os.mkdir(os.path.join(args.path, './results', args.name, "log"))

    # set logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler = logging.FileHandler(os.path.join(args.path, "results/{}/log/{}.log".format(args.name, time.strftime('%c', time.localtime(time.time())))))
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler())
    args.logger = logger
    
    # set cuda
    if torch.cuda.is_available():
        args.logger.info("running on cuda")
        args.device = torch.device("cuda")
        args.use_cuda = True
    else:
        args.logger.info("running on cpu")
        args.device = torch.device("cpu")
        args.use_cuda = False
    
    args.logger.info("[{}] starts".format(args.name))
    

    # 1. load data

    args.logger.info("loading data...")
    
    sos_idx = 0
    eos_idx = 1
    pad_idx = 2
    max_length = 50
    
    vocab = Vocab(init_token='<sos>', eos_token='<eos>', pad_token='<pad>', unk_token='<unk>')
    vocab.load(os.path.join(args.path, args.data_dir, 'vocab.en'))
    vocab_size = len(vocab)
    
    args.sos_idx = sos_idx
    args.eos_idx = eos_idx
    args.pad_idx = pad_idx
    args.max_target_length = max_length
    args.vocab_size = vocab_size
    
    train_loaders, val_loaders, test_loader = get_k_fold_loader(args, vocab)

    # 2. setup
    
    args.logger.info("setting up...")
    model = SAT(args)
    model.to(args.device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=args.pad_idx)
    optimizer = optim.Adamax(model.parameters(), lr=args.learning_rate, weight_decay=5e-5)
    lr_lambda = lambda epoch: min(1.0, (epoch + 1) / args.warmup) 
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    if args.load:
    	loaded_data = load(args, args.ckpt)
    	model.load_state_dict(loaded_data['model'])
#     	optimizer.load_state_dict(loaded_data['optimizer'])

    # 3. train / test
    
    if not args.test:
        # train
        args.logger.info("starting training")
        train_loss_meter = AverageMeter(args, name="Loss", save_all=True, x_label="epoch")
        steps = 1
        for epoch in range(1, 1 + args.epochs * args.k):
            if args.start_from_step is not None:
                if steps < args.start_from_step:
                    optimizer.zero_grad()
                    optimizer.step()
                    scheduler.step()
                    steps += 1
                    continue
            spent_time = time.time()
            model.train()
            train_loss_tmp_meter = AverageMeter(args)
            train_loader, val_loader = train_loaders[(epoch - 1) % args.k], val_loaders[(epoch - 1) % args.k]
            for images, captions in tqdm(train_loader):
                optimizer.zero_grad()
                images, captions = preprocess(args, images, captions, vocab)
                batch_size = captions.shape
                preds, attns = model(images.to(args.device), batch_size[1] - 1, answers=captions[:, :-1].to(args.device))
#                 print(preds.shape)
                loss = loss_fn(preds.view(batch_size[0] * (batch_size[1] - 1), args.vocab_size), captions[:, 1:].flatten(0).to(args.device))
                loss.backward()
                optimizer.step()
                train_loss_tmp_meter.update(loss, weight=batch_size[0])
                steps += 1
            scheduler.step()       
            train_loss_meter.update(train_loss_tmp_meter.avg)
            train_loss_meter.plot(scatter=False)
            spent_time = time.time() - spent_time
            args.logger.info("[{}] train loss: {:.3f} took {:.1f} seconds".format(epoch, train_loss_tmp_meter.avg, spent_time))
             
            # validate and save
            if steps % args.save_period == 0:
                save(args, "epoch_{}".format(steps), {'model': model.state_dict()})
                model.eval()
                pred_list = []
                answer_list = []
                with torch.no_grad():
                    for images, captions in tqdm(val_loader):
                        optimizer.zero_grad()
                        batch_size = len(images)
                        preds, attns = model.infer(images.to(args.device))
                        preds = preds.repeat(5, 1)
                        pred_list += seq2sen(preds.cpu().numpy().tolist(), vocab)
                        for seqs in captions:
                            answer_list += [*seqs]
                with open('results/pred_val.txt', 'w', encoding='utf-8') as f:
                    for line in pred_list:
                        f.write('{}\n'.format(line))
                with open('results/answer_val.txt', 'w', encoding='utf-8') as f:
                    for line in answer_list:
                        f.write('{}\n'.format(line))
        os.system('bash scripts/bleu.sh results/pred_val.txt results/answer_val.txt') 
    else:
        pass
        # test
        args.logger.info("starting test")
        model.eval()
        pred_list = []
        answer_list = []
        with torch.no_grad():
            for images, captions in tqdm(test_loader):
                optimizer.zero_grad()
                batch_size = len(images)
                preds, attns = model.infer(images.to(args.device))
                preds = preds.repeat(5, 1)
                pred_list += seq2sen(preds.cpu().numpy().tolist(), vocab)
                for seqs in captions:
                    answer_list += [*seqs]
        with open('results/pred_test.txt', 'w', encoding='utf-8') as f:
            for line in pred_list:
                f.write('{}\n'.format(line))
        with open('results/answer_test.txt', 'w', encoding='utf-8') as f:
            for line in answer_list:
                f.write('{}\n'.format(line))
        os.system('bash scripts/bleu.sh results/pred_test.txt results/answer_test.txt') 
        

if __name__ == '__main__':
    # set args
    parser = argparse.ArgumentParser(description='Glow')
    parser.add_argument(
        '--path',
        type=str,
        default='.')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='dataset')
    parser.add_argument(
        '--result_dir',
        type=str,
        default='results')
    parser.add_argument(
        '--ckpt_dir',
        type=str,
        default="ckpt")
    parser.add_argument(
        '--epochs',
        type=int,
        default=100)
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-5)
    parser.add_argument(
    	'--warmup',
    	type=int,
    	default=5),
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64)
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4)
    parser.add_argument(
        '--test',
        action='store_true')
    parser.add_argument(
        '--save_period',
        type=int,
        default=5)
    parser.add_argument(
        '--start_from_step',
        type=int,
        default=None)
    parser.add_argument(
        '--name',
        type=str,
        default="train")
    parser.add_argument(
        '--ckpt',
        type=str,
        default='_')
    parser.add_argument(
        '--load',
        action='store_true')
    parser.add_argument(
        '--k',
        type=int,
        default=5)
    parser.add_argument(
        '--embedding_dim',
        type=int,
        default=512)
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=512)
    parser.add_argument(
        '--word_dim',
        type=int,
        default=512)
    parser.add_argument(
        '--attn_dim',
        type=int,
        default=512)
    parser.add_argument(
        '--test_data_portion',
        type=float,
        default=0.2)
    
    args = parser.parse_args()

        
    main(args)