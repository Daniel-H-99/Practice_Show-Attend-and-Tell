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
from utils import *
from model import SAT
from tqdm import tqdm
import logging
import time
from PIL import Image
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
import random 
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
    
    args.sos_idx = sos_idx
    args.eos_idx = eos_idx
    args.pad_idx = pad_idx
    args.max_target_length = max_length

    
    train_loaders, val_loaders, test_loader, vocab = get_k_fold_loader(args)
    vocab_size = len(vocab)
    args.vocab_size = vocab_size

    # 2. setup
    
    args.logger.info("setting up...")
    model = SAT(args)
    model.to(args.device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=args.pad_idx)
    optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate)
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
            for k in range(5):
                if args.start_from_step is not None:
                    if steps < 1:
                        optimizer.zero_grad()
                        optimizer.step()
                        scheduler.step()
                        steps += 1
                        continue

                spent_time = time.time()
                model.train()
                train_loss_tmp_meter = AverageMeter(args)
                train_loader, val_loader = train_loaders[0], val_loaders[0]
                for images, captions in tqdm(train_loader):
                    optimizer.zero_grad()
                    images, answers = preprocess(args, k, images, captions, vocab)     
                    batch_size = answers.shape
                    preds, attns = model(images.to(args.device), batch_size[1] - 1, answers=answers[:, :-1].to(args.device))
                    preds = preds / args.temperature
                    loss = loss_fn(preds.view(batch_size[0] * (batch_size[1] - 1), args.vocab_size), answers[:, 1:].flatten(0).to(args.device))
                    loss += args.penalty * ((1 - attns.sum(dim=1)) ** 2).sum(dim=1).mean()
                    loss.backward()
                    optimizer.step()
                    train_loss_tmp_meter.update(loss, weight=batch_size[0])    
                scheduler.step()       
                train_loss_meter.update(train_loss_tmp_meter.avg)
                train_loss_meter.plot(scatter=False)
                spent_time = time.time() - spent_time
                args.logger.info("[{}] train loss: {:.3f} took {:.1f} seconds".format(steps, train_loss_tmp_meter.avg, spent_time))
                
                # validate and save
                if steps % args.save_period == 0:
                    save(args, "epoch_{}".format(steps), {'model': model.state_dict()})
                    model.eval()
                    pred_list = []
                    answer_list = []
                    with torch.no_grad():
                        for images, captions in tqdm(val_loader):
                            images, _ = preprocess(args, 0, images, captions, vocab)
                            captions = [preprocess(args, i, images, captions, vocab)[1].cpu().tolist() for i in range(5)]
                            batch_size = len(images)
                            answers = []
                            for i in range(batch_size):
                                answers.append([captions[j][i] for j in range(5)])
                            answers = [seq2tok(answer, vocab) for answer in answers]
                            preds, attns = model.infer(images.to(args.device))
                            preds = seq2tok(preds.cpu().numpy().tolist(), vocab)
                    print(pred_list)
                    print(answer_list)  
                    print('BLEU-1: %f' % corpus_bleu(answers, preds, weights=(1.0, 0, 0, 0)))
                    print('BLEU-2: %f' % corpus_bleu(answers, preds, weights=(0.5 , 0.5, 0, 0)))
                    print('BLEU-3: %f' % corpus_bleu(answers, preds, weights=(0.3 , 0.3, 0.3, 0)))
                    print('BLEU-4: %f' % corpus_bleu(answers, preds, weights=(0.25, 0.25, 0.25, 0.25)))                
                steps += 1 
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

            with open(os.path.join(args.path, args.result_dir, 'pred_test.txt'), 'w', encoding='utf-8') as f:
                for line in pred_list:
                    f.write('{}\n'.format(line))
            with open(os.path.join(args.path, args.result_dir, 'answer_test.txt'), 'w', encoding='utf-8') as f:
                for line in answer_list:
                    f.write('{}\n'.format(line))
        os.system('bash {} {} {}'.format(os.path.join(args.path, 'scripts/bleu.sh'), os.path.join(args.path, args.result_dir, args.name, 'pred_test.txt'), os.path.join(args.path, args.result_dir, args.name, 'answer_test.txt')))
        

if __name__ == '__main__':
    # set args
    parser = argparse.ArgumentParser(description='SAT')
    parser.add_argument(
        '--path',
        type=str,
        default=".")
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
        default=1e-4)
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
        default=1)
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
        '--temperature',
        type=float,
        default=1)
    parser.add_argument(
        '--penalty',
        type=float,
        default=1)
    parser.add_argument(
        '--test_data_portion',
        type=float,
        default=0.2)
    
    args = parser.parse_args()

        
    main(args)