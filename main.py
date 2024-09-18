import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from models.Audio_model import Audio_Frontend, AudioModel_pre_Cnn10
from models.model_zoo.panns import PANNS_Cnn10
from datasets.uffia import get_dataloader
from tqdm import tqdm

# Training parameters
batch_size = 300
max_epoch = 300
learning_rate = 1e-3
seed = 25
classes_num = 4

# Audio feature parameters
sample_rate = 64000
window_size = 2048
hop_size = 1024
mel_bins = 128
fmin = 1
fmax = 128000

# Set random seed
torch.manual_seed(seed)
np.random.seed(seed)

# Initialize data loaders
train_loader = get_dataloader(split='train', batch_size=batch_size, sample_rate=sample_rate, shuffle=True, seed=seed, drop_last=True)
val_loader = get_dataloader(split='val', batch_size=batch_size, sample_rate=sample_rate, seed=seed, drop_last=True)

# Initialize model
audio_frontend = Audio_Frontend(sample_rate, window_size, hop_size, mel_bins, fmin, fmax)
audio_encoder = PANNS_Cnn10(classes_num=4)
model = AudioModel_pre_Cnn10(audio_frontend, audio_encoder)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

# Training loop
for epoch in range(max_epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(train_loader, "Training:"):
        inputs = batch['waveform'].to(device)
        targets = batch['target'].to(device)

        # Forward pass
        outputs = model(inputs)['clipwise_output']
        loss = criterion(outputs, targets.argmax(dim=1))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets.argmax(dim=1)).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['waveform'].to(device)
            targets = batch['target'].to(device)

            outputs = model(inputs)['clipwise_output']
            loss = criterion(outputs, targets.argmax(dim=1))

            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets.argmax(dim=1)).sum().item()

    val_loss /= val_total
    val_acc = val_correct / val_total

    print(f'Epoch [{epoch+1}/{max_epoch}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, '
          f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')

# Save the final model
torch.save(model.state_dict(), 'audio_model.pth')
