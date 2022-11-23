import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __getitem__(self, index):
        x = self.data[index]
        
        return x
    
    def __len__(self):
        return len(self.data)