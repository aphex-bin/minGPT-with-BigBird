from collections import OrderedDict
from torchvision import datasets, transforms
from torch.utils.data import Dataset

from minGPT.gpt import *


class ImageDataset(Dataset):
    """
    wrap up the pytorch MNIST dataset into our own, which will convert images into sequences of integers
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
        a = 255*x//16
        return a[:-1], a[1:] # always just predict the next one in the sequence

def main():
    # initialize model
    hyper_params = HyperParameters()
    hyper_params.n_epochs = 74
    print("Running " + str(hyper_params.n_epochs) +  " epochs")
    gpt = GPT(hyper_params)
    gpt_bb = GPT(hyper_params, big_bird=True)

    # initialize data
    train_data = datasets.FashionMNIST("", train=True, download=True, 
                                transform=transforms.Compose([transforms.ToTensor()]))
    test_data = datasets.FashionMNIST("", train=False, download=True, 
                                transform=transforms.Compose([transforms.ToTensor()]))

    train_data = ImageDataset(train_data)
    test_data = ImageDataset(test_data)

    train_data, val_data = torch.utils.data.random_split(train_data, [54000, 6000],
                                                        generator=torch.Generator().manual_seed(0))

    # train
    trainer = Trainer(gpt, train_data, val_data, hyper_params)
    trainer.train("train_extra")

    trainer_bb = Trainer(gpt_bb, train_data, val_data, hyper_params)
    trainer_bb.train("train_bb_extra")


if __name__ == "__main__":
    pass