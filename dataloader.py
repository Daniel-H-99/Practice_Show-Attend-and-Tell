import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader 
from dataset.field import Field

def get_loader(src, tgt, src_vocab, tgt_vocab, batch_size, shuffle=False):
    max_length = 50

    strip_func = lambda x: x[:max_length]
    src_field = Field(src_vocab,
                      preprocessing=None,
                      postprocessing=strip_func)
    tgt_field = Field(tgt_vocab,
                      preprocessing=lambda seq: ['<sos>'] + seq + ['<eos>'],
                      postprocessing=strip_func)

    src = [src_field(seq.split()) for seq in src]
    tgt = [tgt_field(seq.split()) for seq in tgt]

    data_loader = DataLoader(src, tgt, batch_size=batch_size, pad_idx=2, shuffle=shuffle)

    return data_loader

def pad(batch, pad_idx):
    max_len = 0
    for seq in batch:
        if max_len < len(seq):
            max_len = len(seq)

    for i in range(len(batch)):
        batch[i] += [pad_idx] * (max_len - len(batch[i]))

    return batch

# read csv file and make dataloader (a batch is single list of tensors)
def get_k_fold_loader(args, vocab):
    k = args.k
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset = torchvision.datasets.Flickr8k(root=os.path.join(args.path, args.data_dir, "Images"), ann_file=os.path.join(args.path, args.data_dir, "ann.html"), transform=image_transform)
    num_data = len(dataset)
    indices = list(range(num_data))
    np.random.shuffle(indices)

    split = int(np.floor(args.test_data_portion * num_data))
    train_idx, test_idx = indices[split:], indices[:split]
    
    test_sampler = SubsetRandomSampler(test_idx)
    test_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.num_workers, drop_last=True)
    
    train_loaders = []
    val_loaders = []
    num_val = len(train_idx) // k
    for _ in range(k):
        train_idx, val_idx = train_idx[0: k * num_val] + train_idx[(k + 1) * num_val:], train_idx[k * num_val:(k + 1) * num_val]
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_loaders.append(DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler,
                              num_workers=args.num_workers, drop_last=True))
        val_loaders.append(DataLoader(dataset, batch_size=args.batch_size, sampler=val_sampler,
                              num_workers=args.num_workers, drop_last=True))
    return train_loaders, val_loaders, test_loader

def preprocess(args, images, captions, vocab):
    k = args.k
    max_length = args.max_target_length
    captions = captions[k % 5]
    strip_func = lambda x: x[:max_length]
    captions_field = Field(vocab,
                           preprocessing=lambda seq: ['<sos>'] + seq + ['<eos>'],
                           postprocessing=strip_func)
    captions = [captions_field(caption.split()) for caption in captions]
    captions = pad(captions, args.pad_idx)
    captions = torch.Tensor(captions).long()
    return images, captions  

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
        captions = captions[k % 5]
        strip_func = lambda x: x[:self.max_length]
        captions_field = Field(self.vocab,
                               preprocessing=lambda seq: ['<sos>'] + seq + ['<eos>'],
                               postprocessing=strip_func)
        captions = [captions_field(caption.split()) for caption in captions]
        captions = pad(captions)
        captions = torch.Tensor(captions).long()
        print(captions.shape)
        return images, captions
        