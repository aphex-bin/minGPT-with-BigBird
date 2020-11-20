import numpy as np
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm

import torch
from torch.nn import *
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset

from minGPT.gpt import *


class ImageDataset(Dataset):
    """
    wrap up the pytorch f-MNIST dataset into sequences of integers, with class as target
    """
    
    def __init__(self, pt_dataset):
        self.pt_dataset = pt_dataset
        self.block_size = 28*28 - 1
        
    def __len__(self):
        return len(self.pt_dataset)

    def __getitem__(self, idx):
        x, y = self.pt_dataset[idx]
        x = torch.from_numpy(np.array(x)).view(-1) # flatten out all pixels
        x = x.float() # -> float
        a = 255*x // 16
        # make one hot encoding of y        
        return a[:-1], y    #return the image and the label


class FineTune(Module):
    '''
    fine tune the pretrained GPT network on the image classification task
    '''
    def __init__(self, gpt, hyper_params, learn_rate = 3*1e-4):
        super().__init__()
        self.hyper_params = hyper_params
        self.n_classes = 10
        self.gpt = gpt

        self.big_bird = gpt.big_bird
        
        ### Same parameters as in GPT class
        self.pixel_embedding = gpt.pixel_embedding
        self.position_embeddings = gpt.position_embeddings
        self.decoder_stack = gpt.decoder_stack
        self.layer_norm = gpt.layer_norm
        self.output_projection = gpt.output_projection

        #Put the final head on the transformers
        self.flatten = Flatten()
        self.final_output_projection = Linear(hyper_params.n_pixels*hyper_params.n_colors, self.n_classes)

        self.batchsize = hyper_params.batchsize
        self.learn_rate = learn_rate

    
    def forward(self, x, targets=None):
        # x: [batchsize, n_pixels, n_embd]
        # create embedding with position information
        x = self.pixel_embedding(x) + self.position_embeddings

        # run x through decoder stack
        x = self.decoder_stack(x)
        x = self.layer_norm(x)
        x = self.flatten(x)

        # x: [batchsize, n_pixels, n_embd] (unchanged)
        x = self.final_output_projection(x)

        # run x through final read out
        # prediction: [batchsize, n_classes]
        prediction = functional.softmax(x, dim=-1)

        loss = None
        if targets is not None:
            loss = functional.cross_entropy(x,targets.view(-1))

        return prediction, loss


def main():
    # init data and load GPT model
    hyper_params = HyperParameters(bug_test=False)
    gpt = GPT(hyper_params, big_bird = False)
    gpt_bb = GPT(hyper_params, big_bird = True)

    # fix checkpoint format and load gpt
    checkpoint_prel = torch.load('checkpoints/train_epoch_74.pt')    # change this path to use another checkpoint
    checkpoint = {}
    for key in checkpoint_prel:
        checkpoint[key[7:]] = checkpoint_prel[key]
    state_dict = gpt.load_state_dict(checkpoint)    #load the model

    # fix checkpoint format and load big bird gpt
    checkpoint_prel = torch.load('checkpoints/train_bb_epoch_74.pt') # change this path to use another checkpoint
    checkpoint = {}
    for key in checkpoint_prel:
        checkpoint[key[7:]] = checkpoint_prel[key]
    state_dict_bb = gpt_bb.load_state_dict(checkpoint)  #load the model

    # fewer epochs are needed to train the fine tuned model
    gpt.n_epochs = 10
    gpt_bb.n_epochs = 10

    # data init
    train_data = datasets.FashionMNIST("", train=True, download=True, 
                                transform=transforms.Compose([transforms.ToTensor()]))
    train_data = ImageDataset(train_data)
    train_data, val_data = torch.utils.data.random_split(train_data, [54000, 6000],
                                                     generator=torch.Generator().manual_seed(0))

    # initialize trainers
    trainer = Trainer(FineTune(gpt, hyper_params), train_data, val_data, hyper_params)
    trainer_bb = Trainer(FineTune(gpt_bb, hyper_params), train_data, val_data, hyper_params)

    # train and save
    trainer.train("fine_tune_vanilla")
    trainer_bb.train("fine_tune_bb")
    

if __name__ == "__main__":
    main()
