import data
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

train_data = data.Sled2DDataGenerator("/home/jperez/data/sled350/", transform, 0, 510 + 1)
train_data_loader = DataLoader(train_data, batch_size=2, shuffle=True)

print(next(iter(train_data_loader)))

model = nn.Sequential(nn.Conv2d(1, 1, 128, padding="same"))
print(model)

is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"CUDA is {is_cuda} and {torch.cuda.device_count()} GPUs")
model.to(device)

opt = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
model.train()

step_loss = 0
with tqdm(total=len(train_data_loader), desc="Training") as p_bar:
    for step, (batch_x, batch_y) in enumerate(train_data_loader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        opt.zero_grad()

        y_pred = model(batch_x)
        loss = loss_fn(batch_y, y_pred)

        step_loss += loss
        p_bar.update()
        p_bar.postfix = f"{step_loss/(step+1):.5f}"
