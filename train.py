import os
import torch
from torch.utils.data import DataLoader
import wandb
import mat73
from dataloader import BasicDataset
import numpy as np
import random
from torch.utils.data import DataLoader
from model import Net
import torch.optim as optim
import time
from loss import Euclidean_loss, PSNR
from tqdm import tqdm

# wandb init
wandb.init(project="recon_TMI", job_type="training")
# set random seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(8)


# Load data
train_data = mat73.loadmat('data/DNN4x1_TrVal.mat')
dataset = BasicDataset(train_data)

# Too much data, Random sample
_, dataset = torch.utils.data.random_split(dataset, [21731, 1000])
train_data, valid_data = torch.utils.data.random_split(dataset, [700, 300])

# Data loader
train_data_size = len(train_data)
train_loader = DataLoader(train_data, batch_size=30, shuffle=True)
valid_data_size = len(valid_data)
valid_loader = DataLoader(valid_data, batch_size=30, shuffle=True)

if not os.path.isdir('checkpoints'):
    os.makedirs('checkpoints')

# Load model
model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_ids = [0, 1, 2]
model = torch.nn.DataParallel(model, device_ids=device_ids)
model = model.cuda(device=device_ids[0])

# Weight init
for m in model.modules():
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight, gain=torch.nn.init.calculate_gain('relu'))

# Loss function
# loss_func = nn.MSELoss()
loss_func = Euclidean_loss()
psnr = PSNR()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-5)
epochs = 100

# scheduler
scheduler_lr = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, mode="min", patience=10,
                                                    cooldown=3, min_lr=1e-7, verbose=True)
# Train
best_loss = 0.0
best_epoch = 0
for epoch in range(epochs):
    epoch_start = time.time()
    print("Epoch: {}/{}".format(epoch + 1, epochs))
    model.train()
    train_loss = 0.0
    valid_loss = 0.0
    train_psnr = 0.0
    valid_psnr = 0.0

    for data in tqdm(train_loader):
        inputs = data['image'].to(device)
        labels = data['label'].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        train_loss += loss.item() * inputs.size(0)
        train_psnr += psnr(outputs, labels).item() * inputs.size(0)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        model.eval()
        for j, valid_data in enumerate(valid_loader):
            inputs = valid_data['image'].to(device)
            labels = valid_data['label'].to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            valid_loss += loss.item() * inputs.size(0)
            valid_psnr += psnr(outputs, labels).item() * inputs.size(0)
    # calculate average losses and psnr
    avg_train_loss = train_loss / train_data_size
    avg_valid_loss = valid_loss / valid_data_size

    avg_train_psnr = train_psnr / train_data_size
    avg_valid_psnr = valid_psnr / valid_data_size
    scheduler_lr.step(avg_valid_loss)
    if epoch == 0:
        best_loss = avg_valid_loss
        best_epoch = epoch + 1
    if best_loss > avg_valid_loss:
        best_loss = avg_valid_loss
        best_epoch = epoch + 1

    epoch_end = time.time()
    print(
        "Epoch: {:03d}, lr:{:.3g}, Training: Loss: {:.4f}, Training: PSNR: {:.4f}\n\t\tValidation: Loss: {:.4f}, Validation: PSNR: {:.4f} Time: {:.4f}s".format(
            epoch + 1, optimizer.state_dict()['param_groups'][0]['lr'], avg_train_loss,avg_train_psnr, avg_valid_loss, avg_valid_psnr,
            epoch_end - epoch_start
        ))
    print("Best Loss for validation : {:.4f} at epoch {:03d}".format(best_loss, best_epoch))
    if (epoch + 1) % 10 == 0:
        torch.save(model, 'checkpoints/' + '_checkpoint_' + str(epoch + 1) + '.pt')
    # wandb log
    wandb.log({
        "Epoch": epoch,
        "lr": optimizer.defaults['lr'],
        "Train Loss": avg_train_loss,
        "Train PSNR": avg_train_psnr,
        "Valid Loss": avg_valid_loss,
        "Valid PSNR": avg_valid_psnr})