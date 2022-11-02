from torch.utils.data import Dataset
import torch


class BasicDataset(Dataset):
    def __init__(self, mat_data):
        self.labels = mat_data["images"]["labels"]
        self.data = mat_data["images"]["data"]

    def __len__(self):
        return self.data.shape[2]

    def __getitem__(self, idx):
        img = self.data[:, :, idx]
        label = self.labels[:, :, idx]

        return {
            'image': torch.as_tensor(img.copy()).float().unsqueeze(0).contiguous(),
            'label': torch.as_tensor(label.copy()).float().unsqueeze(0).contiguous()
        }
