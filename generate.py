import numpy as np
import matplotlib.pyplot as plt
import copy
from collections import OrderedDict

import torch
from torch.nn import *
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset

from minGPT.gpt import *


class ImageDataset(Dataset):
    """
    create custom FashionMNIST dataset
    """
    
    def __init__(self, pt_dataset):
        self.pt_dataset = pt_dataset
        self.block_size = 28*28
        
    def __len__(self):
        return len(self.pt_dataset)

    def __getitem__(self, idx):
        x, y = self.pt_dataset[idx]
        x = torch.from_numpy(np.array(x)).view(-1) # flatten out all pixels
        x = x.float() # -> float
        a = 255*x//16
        return a[:-1], y # always just predict the next one in the sequence


class Generator:
    '''
    generate an image using the trained network
    '''
    def __init__(self, gpt, cropped_image, start_idx, deterministic = True):
        self.gpt = gpt
        self.cropped_image = cropped_image
        self.start_idx = start_idx
        self.deterministic = True
        
        self.device = "cpu"
        if torch.cuda.is_available():
            print("Using CUDA")
            self.device = torch.cuda.current_device()
            gpt = torch.nn.DataParallel(gpt).to(self.device)

    def softmax(self, probs):
        normalized_probs = np.zeros(probs.shape)
        epsilon = 0
        for i in range(probs.shape[0]):
            normalized_probs[i] = np.exp(probs[i]+epsilon)/np.sum(np.exp(probs+epsilon))

        for i in range(probs.shape[0]):
            normalized_probs[i] = (normalized_probs[i])/np.sum(normalized_probs)

        return normalized_probs

    def generate(self):
        for i in range(self.start_idx, 782):
            x = self.cropped_image.to(self.device)
            x = x.type(torch.LongTensor)
            x = x.view(x.size()[0], -1)
            x = torch.transpose(x,0,1)
            x = x.view(1,1,783)
            x = x.to(self.device)

            predictions = gpt.forward(x)
            
            # -sample from predictions[i]
            predictions = predictions[0]    #get the torch tensor from the tuple
            predictions = predictions[0,0,i,:].cpu().detach().numpy()
            predictions = np.array(predictions)
            
            if self.deterministic:
                next_pixel = int(np.argmax(predictions))
            else:
                next_pixel = int(np.argmax(np.random.multinomial(1,self.softmax(predictions))))
            self.cropped_image[i + 1] = next_pixel


        return self.cropped_image

def main():
    # initialize data
    train_data = datasets.FashionMNIST("", train=True, download=True,
                                transform=transforms.Compose([transforms.ToTensor()]))
    test_data = datasets.FashionMNIST("", train=False, download=True, 
                                transform=transforms.Compose([transforms.ToTensor()]))

    train_data = ImageDataset(train_data)
    train_set, val_set = torch.utils.data.random_split(train_data, [50000, 10000])

    test_data = ImageDataset(test_data)

    # get one of each clothing item, and crop them half way through (391)
    cropped_images = []
    images = []
    caught_labels = []
    label = 0
    it = 0
    num_examples_per_class = 5

    while True:
        image = test_data.__getitem__(it)[0]
        label = test_data.__getitem__(it)[1]

        if len(np.where(np.array(caught_labels) == label)[0]) < num_examples_per_class:
            images.append(image)
            cropped_image = torch.clone(image)
            cropped_image[391:] = 0
            cropped_images.append(cropped_image)
            caught_labels.append(label)

        it += 1
        if len(caught_labels) == 50:
            break
    
    # save examples
    for i in range(len(caught_labels)):
        #save the original and the cropped
        original = images[i]
        original = np.concatenate((original, np.array([0])))
        np.save("generated/original_"+str(i)+"_class_"+str(caught_labels[i]),original)
        
        cropped = cropped_images[i]
        cropped = np.concatenate((cropped, np.array([0])))
        np.save("generated/cropped_"+str(i)+"_class_"+str(caught_labels[i]),cropped)

    # initialize gpt
    hyper_params = HyperParameters()
    gpt = GPT(hyper_params, big_bird = False)
    state_dict = torch.load('checkpoints/train_epoch_74.pt') # change this path to use another checkpoint
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # removes `module.` from the end of the string
        new_state_dict[name] = v
    gpt.load_state_dict(new_state_dict)                #load the model

    # initialize big bird gpt
    hyper_params = HyperParameters()
    gpt_bb = GPT(hyper_params, big_bird = True)
    state_dict = torch.load('checkpoints/train_bb_epoch_74.pt') # change this path to use another checkpoint
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # removes `module.` from the end of the string
        new_state_dict[name] = v
    gpt_bb.load_state_dict(new_state_dict)   

    # generate and save examples
    for i in range(len(caught_labels)):
        print(i)
        generated = Generator(gpt,cropped_images[i],391)
        generated = generated.generate()
        generated = np.concatenate((generated, np.array([0])))
        np.save("generated/generated_"+str(i)+"_class_"+str(caught_labels[i]),generated)
        
        bb_generated = Generator(gpt_bb,cropped_images[i],391)
        bb_generated = bb_generated.generate()
        bb_generated = np.concatenate((bb_generated, np.array([0])))
        np.save("generated/bb_generated_"+str(i)+"_class_"+str(caught_labels[i]),bb_generated)


if __name__ == "__main__":
    pass
