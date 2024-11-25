from torch.utils.data import Dataset, DataLoader, random_split
import torch
import numpy as np


class MyData(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=float).reshape(-1, 1)
        self.y = torch.tensor(y, dtype=float).reshape(-1, 1)

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.y)


if __name__ == '__main__':
    SEED = 42
    np.random.seed(SEED)
    X = np.random.rand(100)
    y = 2.3 + 1.2 * X + np.random.randn(100) * 0.1

    dataset = MyData(X, y)
    train_data, val_data = random_split(dataset, [.85, .15],
                                        generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)
