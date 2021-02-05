import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, Dataset
from dataset.field import Field, Vocab
from utils import save, load
from typing import Any, Callable, Dict, List, Optional, Tuple
from torchvision.datasets.vision import VisionDataset 
from collections import defaultdict
from PIL import Image
import random 
import matplotlib.pyplot as plt

def pad(batch, pad_idx):
    max_len = 0
    for seq in batch:
        if max_len < len(seq):
            max_len = len(seq)

    for i in range(len(batch)):
        batch[i] += [pad_idx] * (max_len - len(batch[i]))

    return batch

# read csv file and make dataloader (a batch is single list of tensors)
def get_k_fold_loader(args):
    k = args.k
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset = Flickr8k(args, root=os.path.join(args.path, args.data_dir, "Images"), ann_file=os.path.join(args.path, args.data_dir, "ann.pt"), transform=image_transform)
    vocab = dataset.get_vocab()
    num_data = len(dataset)
    indices = list(range(num_data))
    np.random.shuffle(indices)

    split = int(np.floor(args.test_data_portion * num_data))
    remain_idx, test_idx = indices[split:], indices[:split]
    
    test_sampler = SubsetRandomSampler(test_idx)
    test_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.num_workers, drop_last=True)
    
    train_loaders = []
    val_loaders = []
    num_val = len(remain_idx) // k
    for i in range(k):
        train_idx, val_idx = remain_idx[0: i * num_val] + remain_idx[(i + 1) * num_val:], remain_idx[i * num_val:(i + 1) * num_val]
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_loaders.append(DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler,
                              num_workers=args.num_workers, drop_last=True))
        val_loaders.append(DataLoader(dataset, batch_size=args.batch_size, sampler=val_sampler,
                              num_workers=args.num_workers, drop_last=True))
    return train_loaders, val_loaders, test_loader, vocab

def preprocess(args, k, images, captions, vocab):
    max_length = args.max_target_length
    answers = captions[k % 5]
    strip_func = lambda x: x[:max_length]
    answers_field = Field(vocab,
                           preprocessing=lambda seq: ['<sos>'] + seq + ['<eos>'],
                           postprocessing=strip_func)
    answers = [answers_field(answer.split()) for answer in answers]
    answers = pad(answers, args.pad_idx)
    answers = torch.Tensor(answers).long()

    return images, answers

class KFoldFlickrLoader(DataLoader):
    def __init__(self, k, max_len, vocab, *args, **kwargs):
        super(KFoldFlickrLoader, self).__init__(*args, **kwargs)
        self.k = k
        self.max_len = max_len
        self.vocab = vocab
    def __next__(self):
        print("called")
        images, captions = super(KFoldFlickrLoader, self).__next__()
        print(captions)
        captions = captions[self.k % 5]
        strip_func = lambda x: x[:self.max_length]
        captions_field = Field(self.vocab,
                               preprocessing=lambda seq: ['<sos>'] + seq + ['<eos>'],
                               postprocessing=strip_func)
        captions = [captions_field(caption.split()) for caption in captions]
        captions = pad(captions)
        captions = torch.Tensor(captions).long()
        print(captions.shape)
        return images, captions
        
class Flickr8k(VisionDataset):
    def __init__(
            self,
            args,
            root: str,
            ann_file: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        super(Flickr8k, self).__init__(root, transform=transform,
                                       target_transform=target_transform)
        self.annotations = torch.load(ann_file, map_location='cpu')
        self.ids = list(sorted(self.annotations.keys()))
        self.vocab = Vocab(init_token='<sos>', eos_token='<eos>', pad_token='<pad>', unk_token='<unk>')
        sentences = []
        for key in self.annotations.keys():
            sentences += self.annotations[key]
        self.vocab.build_vocab(sentences)
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img_id = self.ids[index]

        # Image
        img = Image.open(img_id).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        # Captions
        target = self.annotations[img_id]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self) -> int:
        return len(self.ids)

    def get_vocab(self):
        return self.vocab