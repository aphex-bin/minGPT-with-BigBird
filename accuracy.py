import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset

from linearprobe import linearProbe
from finetune import FineTune
from minGPT.gpt import *

class Accuracy:
    """
    Loads supplied trained network and suppies methods for finding
    accuracy on supplied data set. Specifically designed for our
    experiments, not suitable for general use
    """

    def __init__(self, checkpoint_path, test_set, big_bird, class_task, lp, layer, original_model, fine_tuned=False):
        """
        checkpoint path : path to saved pytorch network (string)
        test_set : the data set to measure accuracy on
        big_bird : if network implements the big bird mask (boolean)
        class_task : if the network is to be evaluated on the classification task
        lp : if the network is a linear probe (boolean)
        layer : amount of layers used if network is linear probe (int)
        original_model: if linear probe or fine tune, supply the orignal pytorch model
        fine_tuned : if the network is a fine tuned model (boolean)
        """
        self.class_task = class_task
        self.test_dataset = test_set
        self.lp = lp

        self.hyper_params = HyperParameters()

        self.network = GPT(self.hyper_params, big_bird=big_bird)

        if fine_tuned:
            self.network = FineTune(self.network, self.hyper_params)

        # if not linear probe experiment, checkpoint_path and original model is same
        if not self.lp:
            original_model = checkpoint_path
        print("Class",original_model)
        state_dict = self.import_model(original_model)

        self.network.load_state_dict(state_dict)  # load the model

        # if linear probe experiment, load base model and then add lp layer and load checkpoint
        if self.lp:
            self.network = linearProbe(self.network, layer, self.hyper_params)
            lp_dict = self.import_model(checkpoint_path)
            self.network.output_projection.weight = torch.nn.Parameter(lp_dict["output_projection.weight"])
            self.network.output_projection.bias = torch.nn.Parameter(lp_dict["output_projection.bias"])


        # default to CPU, but check if GPU available
        self.device = "cpu"

        if torch.cuda.is_available():
            print("Using CUDA")
            self.device = torch.cuda.current_device()
            self.network = torch.nn.DataParallel(self.network).to(self.device)

    def import_model(self,checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        return new_state_dict

    def class_acc(self, predictions, target):
        numpy_pred = np.argmax(predictions.detach().cpu().numpy(), axis=1).reshape((16))
        numpy_target = target.cpu().numpy().reshape((16))

        acc = len(np.where(numpy_pred == numpy_target)[0]) / 16

        return acc, np.nan

    def recon_acc(self,predictions, target):
        numpy_pred = predictions.detach().cpu().numpy()
        numpy_pred = np.argmax(numpy_pred, axis=2).astype('float64')

        numpy_target = target.cpu().numpy().astype('float64')

        numpy_target /= np.max(numpy_target)
        numpy_pred /= np.max(numpy_pred)

        err = np.abs(numpy_target - numpy_pred)
        err_std = np.std(err)
        err = np.mean(err)
        acc = 1 - err

        return acc, err_std

    def get_accuracy(self, test_name):
        learn_rate = self.hyper_params.learn_rate
        betas = self.hyper_params.betas
        accs = []
        stds = []

        optimizer = torch.optim.Adam(self.network.parameters(), lr=learn_rate,
                                     betas=betas, weight_decay=0)

        def run_epoch():
            self.network.train(False)
            data = self.test_dataset

            print("Initialising dataloader with batch size:")
            print(self.hyper_params.batchsize)
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=self.hyper_params.batchsize,
                                num_workers=self.hyper_params.n_workers)
            losses = []

            print("Done, looping through epoch...")
            pbar = tqdm(enumerate(loader), total=len(loader))

            for it, (x, target) in pbar:

                x = x.to(self.device)
                x = x.type(torch.LongTensor)
                x = x.view(x.size()[0], -1)

                target = target.to(self.device)
                target = target.type(torch.LongTensor)
                target = target.view(target.size()[0], -1)

                # forward-pass
                with torch.set_grad_enabled(False):
                    predictions, loss = self.network(x, target)
                    loss = loss.mean()  # if losses are scattered on multiple gpus find mean
                    losses.append(loss.item())

                if self.class_task:
                    acc, err_std = self.class_acc(predictions, target)
                else:
                    acc, err_std = self.recon_acc(predictions, target)

                accs.append(acc)
                stds.append(err_std)
                pbar.set_description(f"epoch {0 + 1} iter {it}: train loss {loss.item():.5f}")

            val_loss = float(np.mean(losses))
            acc = np.mean(np.array(accs))
            std = np.mean(np.array(stds))
            return val_loss, acc, std

        print("Starting epoch loop over " + str(1) + " epoch")

        test_loss, accuracy, std = run_epoch()

        np.save("accuracy/" + test_name + "_test_loss", np.array([test_loss]))
        np.save("accuracy/" + test_name + "_test_acc", np.array([accuracy]))
        np.save("accuracy/" + test_name + "_test_acc_std", np.array([std]))

        print("loss on test is " + str(test_loss))
        print("accuracy on test is " + str(accuracy))
        print("std of acc on test is" + str(std))


class ClassificationDataset(Dataset):
    """
    wrap up the pytorch f-MNIST dataset into sequences of integers, with class as target
    """

    def __init__(self, pt_dataset):
        self.pt_dataset = pt_dataset
        self.block_size = 28 * 28 - 1

    def __len__(self):
        return len(self.pt_dataset)

    def __getitem__(self, idx):
        x, y = self.pt_dataset[idx]
        x = torch.from_numpy(np.array(x)).view(-1)  # flatten out all pixels
        x = x.float()  # -> float
        a = 255 * x // 16
        # make one hot encoding of y
        return a[:-1], y  # return the image and the label


class ReconstructionDataset(Dataset):
    """
    wrap up the pytorch f-MNIST dataset into sequences of integers, with next pixel as target
    """

    def __init__(self, pt_dataset):
        self.pt_dataset = pt_dataset
        self.block_size = 28 * 28 - 1

    def __len__(self):
        return len(self.pt_dataset)

    def __getitem__(self, idx):
        x, y = self.pt_dataset[idx]
        x = torch.from_numpy(np.array(x)).view(-1)  # flatten out all pixels
        x = x.float()  # -> float
        a = 255 * x // 16
        return a[:-1], a[1:]  # always just predict the next one in the sequence


def accuracy_loop(path_bb_tuples, test_data, class_task, lp=False,
                  original_model=None, fine_tuned=False):
    """
    Called by main. Makes an instance of the Accuracy class on the specified model
    """
    for layer, model_tuple in enumerate(path_bb_tuples):
        name, big_bird = model_tuple
        accuracy_find = Accuracy("checkpoints/" + name + ".pt", test_data,
                                 big_bird, class_task=class_task, lp=lp, layer=layer,
                                 original_model = original_model, fine_tuned=fine_tuned)
        accuracy_find.get_accuracy(name)


def main(recon_untuned, lp_van_untuned, lp_bb_untuned, recon_tuned,
         lp_van_tuned, lp_bb_tuned):

    # initialize data
    test_data = datasets.FashionMNIST("", train=False, download=True,
                                      transform=transforms.Compose([transforms.ToTensor()]))
    test_data_recon = ReconstructionDataset(test_data)
    test_data_class = ClassificationDataset(test_data)

    # find accuracy of untuned models
    if recon_untuned:
        path_if_bb_tuples_untuned = [("train_epoch_74", False),
                                     ("train_bb_epoch_74", True)]

        print("Finding acc on test data for untuned models")
        accuracy_loop(path_if_bb_tuples_untuned, test_data_recon, class_task=False)

    # find accuracy of linear probe on untuned vanilla models
    if lp_van_untuned:
        original_model = "checkpoints/train_epoch_74.pt" # change this path to use another checkpoint
        path_not_bb_tuples_lp_untuned = [("norm_linear_probe_0_layers_epoch_3", False),
                                         ("norm_linear_probe_1_layers_epoch_3", False),
                                         ("norm_linear_probe_2_layers_epoch_2", False),
                                         ("norm_linear_probe_3_layers_epoch_1", False),
                                         ("norm_linear_probe_4_layers_epoch_1", False),
                                         ("norm_linear_probe_5_layers_epoch_0", False),
                                         ("norm_linear_probe_6_layers_epoch_0", False),
                                         ("norm_linear_probe_7_layers_epoch_1", False),]

        print("Finding acc on test data for linear probes on untuned van models")
        accuracy_loop(path_not_bb_tuples_lp_untuned,test_data_class, class_task=True, lp=True,
                      original_model = original_model)

    # find accuracy of linear probe on untuned big bird models
    if lp_bb_untuned:
        original_model = "checkpoints/train_bb_epoch_74.pt" # change this path to use another checkpoint
        path_is_bb_tuples_lp_untuned = [("norm_linear_probe_bb_0_layers_epoch_3", True),
                                        ("norm_linear_probe_bb_1_layers_epoch_2", True),
                                        ("norm_linear_probe_bb_2_layers_epoch_2", True),
                                        ("norm_linear_probe_bb_3_layers_epoch_1", True),
                                        ("norm_linear_probe_bb_4_layers_epoch_0", True),
                                        ("norm_linear_probe_bb_5_layers_epoch_1", True),
                                        ("norm_linear_probe_bb_6_layers_epoch_1", True),
                                        ("norm_linear_probe_bb_7_layers_epoch_0", True),]

        print("Finding acc on test data for linear probes on untuned bb models")
        accuracy_loop(path_is_bb_tuples_lp_untuned,test_data_class, class_task=True, lp=True,
                        original_model = original_model)

    # find accuracy of fine tuned models:
    if recon_tuned:
        path_if_bb_tuples_finetuned = [("fine_tune_vanilla_epoch_1", False),
                                     ("fine_tune_bb_epoch_2", True)]
        print("Finding acc on test data for fine tuned models")
        accuracy_loop(path_if_bb_tuples_finetuned,test_data_class, class_task=True, fine_tuned=True)

    # find accuracy of linear probe on tuned vanilla models
    if lp_van_tuned:
        original_model = "checkpoints/fine_tune_vanilla_epoch_1.pt" # change this path to use another checkpoint
        path_not_bb_tuples_lp_finetuned = [("norm_fine_tuned_linear_probe_0_layers_epoch_2", False),
                                           ("norm_fine_tuned_linear_probe_1_layers_epoch_2", False),
                                           ("norm_fine_tuned_linear_probe_2_layers_epoch_3", False),
                                           ("norm_fine_tuned_linear_probe_3_layers_epoch_0", False),
                                           ("norm_fine_tuned_linear_probe_4_layers_epoch_0", False),
                                           ("norm_fine_tuned_linear_probe_5_layers_epoch_1", False),
                                           ("norm_fine_tuned_linear_probe_6_layers_epoch_1", False),
                                           ("norm_fine_tuned_linear_probe_7_layers_epoch_1", False),]
        print("Finding acc on test data for lp on fine tuned van. models")
        accuracy_loop(path_not_bb_tuples_lp_finetuned,test_data_class, class_task=True, lp=True,
            original_model = original_model, fine_tuned=True)

    # find accuracy of linear probe on tuned big bird models
    if lp_bb_tuned:
        original_model = "checkpoints/fine_tune_bb_epoch_2.pt" # change this path to use another checkpoint
        path_is_bb_tuples_lp_finetuned = [("norm_fine_tuned_linear_probe_bb_0_layers_epoch_2", True),
                                          ("norm_fine_tuned_linear_probe_bb_1_layers_epoch_2", True),
                                          ("norm_fine_tuned_linear_probe_bb_2_layers_epoch_2", True),
                                          ("norm_fine_tuned_linear_probe_bb_3_layers_epoch_1", True),
                                          ("norm_fine_tuned_linear_probe_bb_4_layers_epoch_1", True),
                                          ("norm_fine_tuned_linear_probe_bb_5_layers_epoch_2", True),
                                          ("norm_fine_tuned_linear_probe_bb_6_layers_epoch_0", True),
                                          ("norm_fine_tuned_linear_probe_bb_7_layers_epoch_1", True),]
        print("Finding acc on test data for lp on fine tuned bb models")
        accuracy_loop(path_is_bb_tuples_lp_finetuned,test_data_class, class_task=True, lp=True,
            original_model = original_model, fine_tuned=True)


if __name__ == "__main__":
    """
    Specify in main which experiments to get accuracy on
    """
    main(recon_untuned=True,
         lp_van_untuned=True,
         lp_bb_untuned=True,
         recon_tuned=True,
         lp_van_tuned=True,
         lp_bb_tuned=True)
