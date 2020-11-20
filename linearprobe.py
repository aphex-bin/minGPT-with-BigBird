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
from finetune import FineTune


class ImageDataset(Dataset):
    """
    create custom FashionMNIST dataset
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
        return a[:-1], y    # return the image and the label


class linearProbe(Module):
    '''
    used to train a head attached to some part of the model
    '''
    def __init__(self, gpt, number_of_layers, hyper_params, learn_rate = 3*1e-4):
        super().__init__()
        self.hyper_params = hyper_params

        # copy and crop gpt layers
        cropped_gpt = copy.deepcopy(gpt)
        cropped_gpt.decoder_stack = cropped_gpt.decoder_stack[:number_of_layers]

        # since optimizer in Trainer class trains all parameters in the model, the don't-train-flag has to be set here
        for param in cropped_gpt.decoder_stack.parameters():
            param.requires_grad = False

        ### Same parameters as in GPT class
        self.pixel_embedding = cropped_gpt.pixel_embedding
        self.position_embeddings = cropped_gpt.position_embeddings

        # don't train embeddings
        self.pixel_embedding.requires_grad = False
        self.position_embeddings.requires_grad = False

        self.big_bird = cropped_gpt.big_bird

        self.decoder_stack = cropped_gpt.decoder_stack
        self.layer_norm = cropped_gpt.layer_norm

        self.n_classes = 10
        self.flatten = Flatten()
        self.output_projection = Linear(hyper_params.n_pixels*hyper_params.n_embd, self.n_classes)

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
        x = self.output_projection(x)

        # run x through final read out
        prediction = functional.softmax(x, dim=-1)

        # prediction [batchsize, n_classes]
        loss = None
        if targets is not None:
            loss = functional.cross_entropy(x,targets.view(-1))

        return prediction, loss 


def main(fine_tuned):
    # model initialization
    hyper_params = HyperParameters(bug_test=False)
    hyper_params.n_epochs = 25
    gpt = GPT(hyper_params, big_bird = False)
    gpt_bb = GPT(hyper_params, big_bird = True)

    # load checkpoints
    checkpoint_prel = torch.load('checkpoints/train_epoch_74.pt') # change this path to use another checkpoint
    checkpoint = {}
    for key in checkpoint_prel:
        checkpoint[key[7:]] = checkpoint_prel[key]
    state_dict = gpt.load_state_dict(checkpoint)                #load the model

    checkpoint_prel = torch.load('checkpoints/train_bb_epoch_74.pt') # change this path to use another checkpoint
    checkpoint = {}
    for key in checkpoint_prel:
        checkpoint[key[7:]] = checkpoint_prel[key]
    state_dict_bb = gpt_bb.load_state_dict(checkpoint)              #load the model
    
    # data initialization
    train_data = datasets.FashionMNIST("", train=True, download=True,
                                transform=transforms.Compose([transforms.ToTensor()]))
    train_data = ImageDataset(train_data)
    train_data, val_data = torch.utils.data.random_split(train_data, [54000, 6000],
                                                     generator=torch.Generator().manual_seed(0))

    # linear probe experiment
    if not fine_tuned:
        # dictionary of cropped models, is this the best way?
        model_dict = {}
        model_dict_bb = {}
        for i in range(hyper_params.n_layers):
            model_dict[i] = linearProbe(gpt, i, hyper_params)
            model_dict_bb[i] = linearProbe(gpt_bb, i, hyper_params)

            hyper_params.learn_rate = hyper_params.learn_rate*2

            # train model
            trainer = Trainer(model_dict[i], train_data, val_data, hyper_params)
            trainer_bb = Trainer(model_dict_bb[i], train_data, val_data, hyper_params)

            trainer.train("linear_probe_%s_layers" % (i, ))
            trainer_bb.train("linear_probe_bb_%s_layers" % (i, ))

    # linear probe experiment (fine tuned networks)
    if fine_tuned:
        # init data and load GPT model
        hyper_params = HyperParameters(bug_test=False)
        hyper_params.n_epochs = 25
        fine_tuned_gpt = FineTune(gpt, hyper_params)
        fine_tuned_gpt_bb = FineTune(gpt_bb, hyper_params)

        # load fine tuned models
        # vanilla
        checkpoint_prel = torch.load('checkpoints/fine_tune_vanilla_epoch_1.pt') # change this path to use another checkpoint
        checkpoint = {}

        for key in checkpoint_prel:
            checkpoint[key[7:]] = checkpoint_prel[key]

        state_dict = fine_tuned_gpt.load_state_dict(checkpoint)              #load the model

        # bigbird
        checkpoint_prel = torch.load('checkpoints/fine_tune_bb_epoch_2.pt') # change this path to use another checkpoint
        checkpoint = {}

        for key in checkpoint_prel:
            checkpoint[key[7:]] = checkpoint_prel[key]

        state_dict_bb = fine_tuned_gpt_bb.load_state_dict(checkpoint)              #load the model

        # dictionary of cropped models, is this the best way?
        fine_tuned_model_dict = {}
        fine_tuned_model_dict_bb = {}
        for i in range(hyper_params.n_layers):
            fine_tuned_model_dict[i] = linearProbe(fine_tuned_gpt, i, hyper_params)
            fine_tuned_model_dict_bb[i] = linearProbe(fine_tuned_gpt_bb, i, hyper_params)

            hyper_params.learn_rate = hyper_params.learn_rate*2

            # train model
            trainer = Trainer(fine_tuned_model_dict[i], train_data, val_data, hyper_params)
            trainer_bb = Trainer(fine_tuned_model_dict_bb[i], train_data, val_data, hyper_params)

            trainer.train("norm_fine_tuned_linear_probe_%s_layers_" % (i, ))
            trainer_bb.train("norm_fine_tuned_linear_probe_bb_%s_layers_" % (i, ))


if __name__ == "__main__":
    '''
    choose if generating using a fine tuned network or not
    '''
    main(fine_tuned=True)
    main(fine_tuned=False)
