#!/scratch/sagar/venv/bin/python3.6
from slf_dataset import SLFDataset
import torch
import os
import numpy as np
from networks.ae import AutoencoderSelu
from tqdm import tqdm
import argparse


if torch.cuda.is_available():
    devices = ['cuda']
else:
    devices = ['cpu']

parser = argparse.ArgumentParser(description='Train autoencoder')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--device', type=str, default='cuda', help='device')
parser.add_argument('--num_workers', type=int, default=5, help='number of workers')
parser.add_argument('--criterion', type=str, default='l1', help='one of l1, l2')
parser.add_argument('--train_data_folder', type=str, default='dataset/train_slf', help='data folder')
parser.add_argument('--validation_data_folder', type=str, default='dataset/valid_slf', help='data folder')
parser.add_argument('--model_path', type=str, default='trained_models/model1.pt', help='model path')
parser.add_argument('--resume', action='store_true', help='resume training')
parser.add_argument('--num_epochs', type=int, default=150, help='number of epochs')
parser.add_argument('--img_size', type=int, default=51, help='image size')

args = parser.parse_args()

print('Preparing dataset')
train_set = SLFDataset(root_dir=args.train_data_folder, sample_size=[0.01, 0.2], img_size=args.img_size)
validation_set = SLFDataset(root_dir=args.validation_data_folder, sample_size=[0.01, 0.2], img_size=args.img_size)

loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

network = AutoencoderSelu().to(args.device)
optimizer  = torch.optim.Adam(network.parameters(), lr=args.lr)

saved_epoch = 0
if args.criterion == 'l1':
    criterion = torch.nn.L1Loss()
elif args.criterion == 'l2':
    criterion = torch.nn.MSELoss()
else:
    raise Exception('Criterion not supported')

if args.resume and os.path.isfile(args.model_path):
    checkpoint = torch.load(args.model_path)
    network.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    saved_epoch = checkpoint['epoch']
    lr = checkpoint['lr']

print('Starting training')
for epoch in range(saved_epoch, args.num_epochs):
    train_losses = []
    valid_losses = []

    # learning rate scheduling
    if epoch>30 and epoch<50:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001
    if epoch>49:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.00002

    for batch in tqdm(loader):
        # Get data
        in_features, t_slf = batch 
        in_features = in_features.to(args.device)
        t_slf = t_slf.to(args.device)
        preds = network(in_features)
        
        loss = criterion(t_slf, preds)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
    
    with torch.no_grad():
        for validation_batch in validation_loader:
            in_features, t_slf = validation_batch
            in_features = in_features.to(args.device)
            t_slf = t_slf.to(args.device)
            preds = network(in_features)
            valid_loss = criterion(t_slf, preds)
            valid_losses.append(valid_loss.item())

    print('epoch: {}, train loss: {}, valid loss: {}'.format(epoch, np.mean(train_losses), np.mean(valid_losses)))

    network.eval().cpu()
    torch.save({'epoch': epoch, 
                'model_state_dict': network.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': np.mean(train_losses),
                'lr': args.lr},
                args.model_path)
    
    network.train().to(args.device)

# Finally, save only the network parameters so that the loading is faster
network.eval().cpu()
torch.save({'model_state_dict': network.state_dict()}, args.model_path)
print('Finished Training')

