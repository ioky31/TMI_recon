import torch
import mat73
from scipy.io import savemat
from dataloader import BasicDataset
from torch.utils.data import DataLoader
from loss import Euclidean_loss, PSNR
from tqdm import tqdm

# Load data
test_data = mat73.loadmat('data/DNN4x1_TestVal.mat')
dataset = BasicDataset(test_data)

# Data loader
test_data_size = len(dataset)
testloader = DataLoader(dataset, batch_size=1, shuffle=True)
model = torch.load('checkpoints/_checkpoint_21.pt')
model.eval()
# Loss function
# loss_func = nn.MSELoss()
loss_func = Euclidean_loss()
psnr = PSNR()

# Load model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_ids = [0, 1, 2]
model = torch.nn.DataParallel(model, device_ids=device_ids)
model = model.cuda(device=device_ids[0])

test_loss = 0.0
test_psnr = 0.0
totle_outputs = torch.zeros(test_data['images']['data'].shape)
# Iterate over data
for i, data in enumerate(tqdm(testloader)):
    inputs = data['image'].to(device)
    labels = data['label'].to(device)
    outputs = model(inputs)
    loss = loss_func(outputs, labels)
    test_loss += loss.item() * inputs.size(0)
    test_psnr += psnr(outputs, labels).item() * inputs.size(0)
    outputs = torch.squeeze(outputs)
    totle_outputs[:, :, i] = outputs.cpu().detach()
avg_test_loss = test_loss / test_data_size
avg_test_psnr = test_psnr / test_data_size
# save outputs
totle_outputs = totle_outputs.numpy()
savemat('test_result.mat', {'test': totle_outputs})
# print result
print('Test Loss: {:.6f} \tTest PSNR: {:.6f}'.format(avg_test_loss, avg_test_psnr))
