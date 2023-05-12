from torch.utils.data import Dataset

class MyDataset(Dataset):
    """
    Implements a custom class to use for dataloading with PyTorch dataloader
    """

    def __init__(self, data):
        """
        Initializing the data
        """

        self.data = data
        
    def __getitem__(self, index):
        """
        Loads and returns a sample from the dataset at the given index 
        """

        x = self.data[index]
        
        return x
    
    def __len__(self):
        """
        Returns the number of samples in the dataset
        """
        return len(self.data)