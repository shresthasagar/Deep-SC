from torch.utils.data import Dataset
import torch
import os
import pandas as pd
import scipy.io
import numpy as np

class SLFDataset(Dataset):
    """SLF loader"""

    def __init__(self, 
                 root_dir, 
                 transform=None,  
                 sample_size=[0.01,0.20], 
                 fixed_size=None, 
                 fixed_mask=False, 
                 no_sampling=False, 
                 img_size=51):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            total_data: Number of data points
            sample_size: range off sampling percentage
            fixed_size: if not none, fixed_size will be used as the sampling size
            fixed_mask: if true, the same mask will be used 
        """
        self.root_dir = root_dir
        self.transform = transform
        self.img_size = img_size
        self.NUM_SAMPLES = int(0.20*self.img_size*self.img_size)
        self.nrow, self.ncol = (self.img_size, self.img_size)
        self.num_examples = len(os.listdir(root_dir))
        self.sampling_rate = sample_size[1]-sample_size[0]
        self.omega_start_point = 1.0 - sample_size[1]
        
        if fixed_size:
            self.sampling_rate = 0
            self.omega_start_point = 1.0 - fixed_size
        
        self.fixed_mask = fixed_mask
        self.no_sampling = no_sampling
        if self.fixed_mask:
            rand = self.sampling_rate*torch.rand(1).item()
            self.bool_mask = torch.FloatTensor(1,self.img_size,self.img_size).uniform_() > (self.omega_start_point+rand)
            self.int_mask = self.bool_mask*torch.ones((1,self.img_size,self.img_size), dtype=torch.float32)
        
    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        filename = os.path.join(self.root_dir, str(idx)+'.pt')
        sample = torch.load(filename)

        if self.no_sampling:
            return sample
        
        if not self.fixed_mask:
            rand = self.sampling_rate*torch.rand(1).item()
            bool_mask = torch.FloatTensor(1,self.img_size,self.img_size).uniform_() > (self.omega_start_point+rand)
            int_mask = bool_mask*torch.ones((1,self.img_size,self.img_size), dtype=torch.float32)
            sampled_slf = sample*bool_mask
        else:
            int_mask = self.int_mask
            sampled_slf = sample*self.bool_mask
        
        return torch.cat((int_mask,sampled_slf), dim=0), sample



class SLFDatasetMatTrue(Dataset):

    def __init__(self, root_dir, csv_file, raw_format=False, transform=None, total_data=None, normalize=False):
        """
        Args:
            csv_file (string): Path to the csv file with params.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.params = pd.read_csv(csv_file)

        self.root_dir = root_dir
        self.transform = transform
        self.nrow, self.ncol = (51, 51)
        self.NUM_SAMPLES = int(0.20*51*51)
        self.num_examples = len(self.params)
        if not total_data is None:
            self.num_examples = total_data
        self.normalize = normalize
        self.raw_format = raw_format

    def __len__(self):
        return self.num_examples
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = os.path.join(self.root_dir,
                                self.params.iloc[idx, 0])
        slf_data = scipy.io.loadmat(filename)

        slf = slf_data['Sc']
        
        # take log 
        if not self.raw_format:
            true_slf = torch.tensor(np.log10(slf + 1e-16), dtype=torch.float32)
        else:
            true_slf = torch.tensor(slf, dtype=torch.float32)
                
        if self.normalize:
            if not self.raw_format:
                true_slf = true_slf/true_slf.min()
            else:
                true_slf = true_slf/(true_slf.max()+1e-16)
        
        return true_slf.unsqueeze(dim=0)


