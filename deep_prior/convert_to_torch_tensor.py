
from tqdm import tqdm, trange
import csv
from slf_dataset import SLFDatasetMatTrue
import os
import torch
import csv
import argparse

parser = argparse.ArgumentParser(description='Convert matlab tensor to torch tensor')
parser.add_argument('--src_img_folder', type=str, default='dataset/train_slf_mat', help='source folder')
parser.add_argument('--src_details_file', type=str, default='dataset/details.csv')
parser.add_argument('--target_img_folder', type=str, default='dataset/train_slf', help='target folder')
args = parser.parse_args()

train_set = SLFDatasetMatTrue(root_dir=args.src_img_folder, 
                                csv_file=args.src_details_file, 
                                raw_format=True, 
                                normalize=False)

if not os.path.isdir(args.target_img_folder):
    os.mkdir(args.target_img_folder)

print(f'length of dataset: {len(train_set)}')
for i in trange(len(train_set)):
    t = train_set[i]
    filename = str(i) + '.pt'
    filename = os.path.join(args.target_img_folder, filename)
    torch.save(t, filename)