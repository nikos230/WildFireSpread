import torch
from torch.utils.data import DataLoader
from unet.model import UNet
from utils.dataset import NetCDFDataset
import torch.optim as optim
import torch.nn as nn
import glob

BATCH_SIZE = 4
LEARNING_RATE = 0.001
EPOCHS = 1
TRAIN_FILE_PATH =''


all_files = glob.glob(TRAIN_FILE_PATH)

train_dataset = NetCDFDataset(all_files, split='train')
validation_dataset = NetCDFDataset(all_files, split='validation')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

in_channels = len(train_dataset.input_vars) + len(train_dataset.static_vars)  
model = UNet(in_channels=in_channels*5, out_channels=1)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)


criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# training loop
for epoch in range(EPOCHS):
    train_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
    print(f'Epoch {epoch} out of {EPOCHS}, Training Loss: {train_loss/len(train_loader)}')


    # validation 
    validation_loss = 0.0
    model.eval()
    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            validation_loss += loss.item()
    print(f'Epoch {epoch} out of {EPOCHS}, Validation Loss {validation_loss/len(validation_loader)}')


torch.save(model.state_dict(), 'unet_model_test.pth')