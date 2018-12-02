'''
    Extend the torch.utils.data.Dataset class to build a GestureDataset class.
'''

import torch.utils.data as data
import torchvision as tv
import numpy as np

class HoursDataset(data.Dataset):

    def __init__(self, X, y):
        super().__init__()
        self.hours = X
        self.y = y

    def __len__(self):
        return len(self.hours)

    def __getitem__(self, index):

        thisentry = self.hours[index]
        temp = thisentry.flatten()
        lab = temp[-4:] #temp[-1]
        feat = temp[:-4]
        return feat, lab