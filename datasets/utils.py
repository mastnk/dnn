from torch.utils.data import Dataset

class DatasetNum(Dataset):
    def __init__( self, dataset, num ):
        self.dataset = dataset
        self.num = num

    def __len__( self ):
        return self.num

    def __getitem__(self, index):
        return self.dataset.__getitem__(index)

