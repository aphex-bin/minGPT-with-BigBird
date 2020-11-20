import numpy as np
from tqdm import tqdm

import torch
from torch.nn import *
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader


# this code is adapted from the minGPT code written by Andrej Karpathy
# https://github.com/karpathy/minGPT


class GPT(Module):
    '''
    the GPT model class
    '''
    def __init__(self, hyper_params, big_bird = False):
        super().__init__()
        self.pixel_embedding = Embedding(hyper_params.n_colors, hyper_params.n_embd,)
        self.position_embeddings = Parameter(torch.zeros(1, hyper_params.n_pixels, hyper_params.n_embd))

        self.big_bird = big_bird

        self.decoder_stack = Sequential(*[Block(hyper_params, big_bird = self.big_bird) for _ in range(hyper_params.n_layers)]) # *[] expands a list to multiple arguments of a function
        self.output_projection = Linear(hyper_params.n_embd, hyper_params.n_colors, bias=False)
        self.layer_norm = LayerNorm(hyper_params.n_embd)
        self.n_colors = hyper_params.n_colors
        self.batchsize = hyper_params.batchsize
        self.n_pixels = hyper_params.n_pixels

    def forward(self, x, targets=None):
        # x: [batchsize, n_pixels, n_embd]
        # create embedding with position information
        x = self.pixel_embedding(x) + self.position_embeddings

        # run x through decoder stack
        x = self.decoder_stack(x)
        x = self.layer_norm(x)

        # run x through final read out
        prediction = self.output_projection(x)

        # prediction [batchsize, n_pixels, n_colors]
        loss = None
        if targets is not None:
            loss = functional.cross_entropy(prediction.view(self.batchsize * self.n_pixels, self.n_colors),
                                            targets.view(-1))

        return prediction, loss


class HyperParameters():
    '''
    a class to store hyperparameters used in the network and training
    '''
    def __init__(self):
        # network parameters
        self.n_embd = 16
        self.n_head = 2
        self.n_latent = self.n_embd // self.n_head  # 8
        self.n_layers = 8
        self.n_colors = 16
        self.n_pixels = 783
        self.n_workers = 8

        # training parameters
        self.n_epochs = 25
        self.batchsize = 16 
        self.learn_rate = 3*1e-4 
        self.betas = (0.9, 0.95)
        self.grad_norm_clip = 1.0

        # BigBird parameters
        self.n_neighbours = 96
        self.n_memory = 128


class MultiHeadSelfAttention(Module):
    '''
    the GPT multi head self attention structure
    '''
    def __init__(self, hyper_params, big_bird=False):
        super().__init__()  # initialize parent class
        n_pixels = hyper_params.n_pixels
        n_embd = hyper_params.n_embd
        n_head = hyper_params.n_head
        n_latent = hyper_params.n_latent

        assert (n_embd // n_head == n_latent), \
            "n_head must equally divide n_embd!"

        self.W_k = Linear(n_latent * n_head, n_embd)  # torch applies the transpose of Linear ( W_k(x) = x @ W_k.T )
        self.W_q = Linear(n_latent * n_head, n_embd)
        self.W_v = Linear(n_latent * n_head, n_embd)

        self.W_0 = Linear(n_embd, n_latent * n_head)  # output projection (combines the 8 heads into 1)

        # mask Q @ K.T. Mask should not be trained by optimizer and should thus be a buffer

        if big_bird:
            # construct mask with numpy
            n_neighbours = hyper_params.n_neighbours
            n_memory = hyper_params.n_memory
            np_mask = np.zeros((n_pixels, n_pixels))
            
            #Window mask
            for i in range(-n_neighbours, n_neighbours + 1):
                np_mask += np.diag(np.ones(n_pixels - abs(i)),k=i)

            #Global mask
            np_mask[:, 0:n_memory] = 1

            #convert to bool
            np_mask = np_mask.astype('bool')

            # convert to 4D tensor (and make lower triangular)
            mask = torch.tril(torch.from_numpy(np_mask)).view(1, 1, n_pixels, n_pixels)

            self.register_buffer("mask", mask)
            
        else:
            # define upper triangular matrix stored as 4D tensor for compatibility
            mask = torch.tril(torch.ones(n_pixels, n_pixels)).view(1, 1, n_pixels, n_pixels)
            self.register_buffer("mask", mask)

    def forward(self, X, hyper_params):
        # pytorch linear treats final dim of data as input dim, other dims preserved
        batchsize = X.shape[0]
        n_pixels = hyper_params.n_pixels
        n_embd = hyper_params.n_embd
        n_head = hyper_params.n_head
        n_latent = hyper_params.n_latent

        K = self.W_k(X)  # = tensor, where tensor[i,k...] = X[i,j,..., :] @ W_k.T
        Q = self.W_q(X)
        V = self.W_v(X)

        # K,Q,V: [batchsize, n_pixels, n_latent*n_head]

        # break stacked heads into separate dims
        K = K.view(batchsize, n_pixels, n_head, n_latent)
        Q = Q.view(batchsize, n_pixels, n_head, n_latent)
        V = V.view(batchsize, n_pixels, n_head, n_latent)

        # K,Q,V: [batchsize, n_pixels, n_head, n_latent]

        # switch dims 1 & 2 to put each K,Q,V - matrix in last two dims
        K = K.transpose(1, 2)
        Q = Q.transpose(1, 2)
        V = V.transpose(1, 2)

        # K,Q,V: [batchsize, n_head, n_pixels, n_latent]

        # transpose of matrix should now act on final dims
        attention_mat = Q @ K.transpose(-2, -1) * (n_latent ** (-1 / 2))

        # attention_mat: [batchsize, n_head, n_pixels, n_pixels]
        attention_mat = attention_mat.masked_fill(self.mask[:, :, :n_pixels, :n_pixels] == 0, float('-inf'))
        attention_mat = functional.softmax(attention_mat, dim=-1)

        Z = attention_mat @ V  # Z: [batchsize, n_head, n_pixels, n_latent]
        Z = Z.transpose(1, 2).contiguous().view(batchsize, n_pixels, n_latent * n_head)

        return Z


class Block(Module):
    '''
    the GPT decoder block structure
    '''
    def __init__(self, hyper_params, big_bird = False):
        super().__init__()

        self.ln1 = LayerNorm(hyper_params.n_embd)
        self.ln2 = LayerNorm(hyper_params.n_embd)
        self.attention = MultiHeadSelfAttention(hyper_params, big_bird = big_bird)
        self.hyper_params = hyper_params

        # 4 seems arbitrary. Maybe we should use 1?
        mlp_dim_factor = 4

        self.mlp = Sequential(
            Linear(hyper_params.n_embd, mlp_dim_factor * hyper_params.n_embd),
            GELU(),
            Linear(mlp_dim_factor * hyper_params.n_embd, hyper_params.n_embd)
        )

    def forward(self, x):
        # First residual connection (attention of layer norm of x is added to x)
        x = x + self.attention(self.ln1(x), self.hyper_params)
        # Second residual connection (MLP projection of layernorm of x is added to the output of x)
        x = x + self.mlp(self.ln2(x))

        return x


class Trainer:
    '''
    the trainer class used to train the GPT model
    '''
    def __init__(self, network, train_data, test_data, hyper_params,
        train_test=True):
        self.network = network
        self.train_dataset = train_data
        self.test_dataset = test_data
        self.hyper_params = hyper_params
        self.train_loss = []
        self.val_loss_list = []
        self.train_test = train_test

        # default to CPU, but check if GPU available
        self.device = "cpu"
        if torch.cuda.is_available():
            print("Using CUDA")
            self.device = torch.cuda.current_device()
            self.network = torch.nn.DataParallel(self.network).to(self.device)

    def save_checkpoint(self, filename):
        torch.save(self.network.state_dict(), "checkpoints/" + filename)

    def train(self,test_name):
        learn_rate = self.hyper_params.learn_rate
        betas = self.hyper_params.betas

        optimizer = torch.optim.Adam(self.network.parameters(), lr=learn_rate,
                                     betas=betas, weight_decay=0)

        def run_epoch(split):
            is_train = (split == "train")
            self.network.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            print("initialising dataloader with batch size:")
            print(self.hyper_params.batchsize)
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=self.hyper_params.batchsize,
                                num_workers=self.hyper_params.n_workers)
            losses = []

            print("done, looping through epoch...")
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (x, target) in pbar:
                
                x = x.to(self.device)
                x = x.type(torch.LongTensor)
                x = x.view(x.size()[0], -1)

                target = target.to(self.device)
                target = target.type(torch.LongTensor)
                target = target.view(target.size()[0], -1)

                # forward-pass
                with torch.set_grad_enabled(is_train):
                    predictions, loss = self.network(x, target)
                    loss = loss.mean()  # if losses are scattered on multiple gpus find mean
                    losses.append(loss.item())

                if is_train:
                    # backward-pass
                    self.network.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(),
                                                   self.hyper_params.grad_norm_clip)
                    # update params
                    optimizer.step()

                    # report progress
                    pbar.set_description(f"epoch {epoch + 1} iter {it}: train loss {loss.item():.5f}")

                    if self.train_test:
                        self.train_loss.append(loss.item())

            if not is_train:
                val_loss = float(np.mean(losses))
                return val_loss

        best_loss = np.inf
        print("starting epoch loop over " + str(self.hyper_params.n_epochs) + " epochs")
        for epoch in range(self.hyper_params.n_epochs):

            run_epoch('train')
            np.save(test_name+"_train_loss", np.array(self.train_loss)) 
           
            if self.test_dataset is not None:
                val_loss = run_epoch('test')
                self.val_loss_list.append(val_loss)
                np.save(test_name+"_val_loss", np.array(self.val_loss_list))
                print("val on test is " + str(val_loss))

            if (self.test_dataset is None) or (val_loss < best_loss):
                best_loss = val_loss
                print("saving...")
                self.save_checkpoint(test_name + "epoch_"+str(epoch+10)+".pt")
            

    def train_linear_probe(self,test_name):
        learn_rate = self.hyper_params.learn_rate
        betas = self.hyper_params.betas
        accs = []

        optimizer = torch.optim.Adam(self.network.parameters(), lr=learn_rate,
                                     betas=betas, weight_decay=0)

        def run_epoch(split):
            is_train = (split == "train")
            self.network.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            print("initialising dataloader with batch size:")
            print(self.hyper_params.batchsize)
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=self.hyper_params.batchsize,
                                num_workers=self.hyper_params.n_workers)
            losses = []

            print("done, looping through epoch...")
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (x, target) in pbar:
                
                x = x.to(self.device)
                x = x.type(torch.LongTensor)
                x = x.view(x.size()[0], -1)

                target = target.to(self.device)
                target = target.type(torch.LongTensor)
                target = target.view(target.size()[0], -1)

                # forward-pass
                with torch.set_grad_enabled(is_train):
                    predictions, loss = self.network(x, target)
                    loss = loss.mean()  # if losses are scattered on multiple gpus find mean
                    losses.append(loss.item())

                if is_train:
                    # backward-pass
                    self.network.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(),
                                                   self.hyper_params.grad_norm_clip)
                    # update params
                    optimizer.step()

                    # report progress
                    pbar.set_description(f"epoch {epoch + 1} iter {it}: train loss {loss.item():.5f}")

                    if self.train_test:
                        self.train_loss.append(loss.item())

                    numpy_pred = np.argmax(predictions.detach().cpu().numpy(), axis=1).reshape((16))
                    numpy_target = target.cpu().numpy().reshape((16))

                    acc = len(np.where(numpy_pred == numpy_target)[0])/16
                    accs.append(acc)

            if not is_train:
                val_loss = float(np.mean(losses))
                acc = np.mean(np.array(accs))
                return val_loss, acc

        best_loss = np.inf
        print("starting epoch loop over " + str(self.hyper_params.n_epochs) + " epochs")
        for epoch in range(self.hyper_params.n_epochs):

            run_epoch('train')
            np.save(test_name+"_train_loss", np.array(self.train_loss)) 
           
            if self.test_dataset is not None:
                val_loss, acc = run_epoch('test')
                self.val_loss_list.append(val_loss)
                np.save(test_name+"_val_loss", np.array(self.val_loss_list))
                print("loss on val is " + str(val_loss))
                print("accuracy on val is " + str(acc))

            if (self.test_dataset is None) or (val_loss < best_loss):
                best_loss = val_loss
                print("saving...")
                self.save_checkpoint(test_name + "_epoch_"+str(epoch)+".pt")

            else:
                break
