import torch
import os
from torchvision.utils import make_grid


class Model:
    def __init__(self, dataset):
        self.dataset = dataset
        self.device = torch.device('cuda')
        self.writer = None

    def show_sequence_images(self, tensor):
        """Tensor should have shape [batch, ch, height, width, time]"""
        reshaped = tensor[0].permute(3, 0, 1, 2)
        # reshaped = (reshaped + 1)/2
        output_grid = make_grid(reshaped, nrow=6, normalize=True, scale_each=True)
        return output_grid

    def show_images(self, tensor):
#         tensor = (tensor + 1) / 2
        tensor=torch.true_divide(tensor+1,2)

        output_grid = make_grid(tensor, nrow=6, normalize=True, scale_each=True)
        return output_grid

    def save_checkpoint(self, path, epoch):
        networks = self.collect_networks()
        optims = self.collect_optims()
        dict = {**networks, **optims}
        for key in dict:
            dict[key] = dict[key].state_dict()
        torch.save(dict, os.path.join(path, 'checkpoint_' + str(epoch) + '.tar'))

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        networks = self.collect_networks()
        optims = self.collect_optims()
        combined = {**networks, **optims}
        for attr_key, check_key in zip(combined, checkpoint):
            print(attr_key, check_key)
            if 'optim' in attr_key:
                combined[attr_key].load_state_dict(checkpoint[check_key])
            else:
                combined[attr_key].load_state_dict(checkpoint[check_key], strict=False)

    def collect_networks(self):
        """All networks should contain network in name'"""
        networks_dict = {}
        for attr in self.__dict__:
            if 'network' in attr:
                networks_dict[attr] = self.__dict__[attr]

        return networks_dict

    def collect_optims(self):
        """Optimisers should contain 'optim'"""
        optims_dict = {}
        for attr in self.__dict__:
            if 'optim' in attr:
                optims_dict[attr] = self.__dict__[attr]

        return optims_dict

    def get_num_params(self):
        networks = self.collect_networks()
        tot = 0
        for n in networks:
            pp = 0
            for p in list(networks[n].parameters()):
                nn = 1
                for s in list(p.size()):
                    nn = nn * s
                pp += nn
            tot += pp
        return tot

    def get_gradients(self, module):
        grads = []
        for param in module.parameters():
            try:
                grads.append(param.grad.view(-1))
            except AttributeError:
                continue
        grads = torch.cat(grads)
        return torch.mean(grads)