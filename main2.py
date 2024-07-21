import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tonic
import tonic.transforms as transforms
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, dataset):
    plt.figure(figsize=(20, 30))
    sns.set(font_scale=0.7)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset.classes)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix', fontsize=16)
    plt.xticks(rotation=90)
    plt.show()

# Print versions
print("Tonic version:", tonic.__version__)
print("SNN-Torch version:", snn.__version__)
print("PyTorch version:", torch.__version__)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Dataset and DataLoader
sensor_size = tonic.datasets.DVSGesture.sensor_size
print("Sensor size:", sensor_size)

transform = transforms.Compose([
    transforms.Downsample(spatial_factor=0.1),
    transforms.ToFrame(sensor_size=(64, 64, 2), time_window=10000),
    tonic.transforms.NumpyAsType(int)
])

trainset = tonic.datasets.DVSGesture(save_to='./data', transform=transform, train=True)
testset = tonic.datasets.DVSGesture(save_to='./data', transform=transform, train=False)

batch_size = 12
trainloader = DataLoader(trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), drop_last=True)
testloader = DataLoader(testset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), drop_last=True)

print('\nTrain dataset')
print("Number of training samples:", len(trainset))
print(f"{len(trainset.classes)} Classes:", trainset.classes)

print('\nTEST dataset')
print("Number of testing samples:", len(testset))
print(f"{len(testset.classes)} Classes:", testset.classes)

# Define SCNN with WTA decoder
class SCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 12, 5)
        self.max1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.1)
        self.lif1 = snn.Leaky(beta=0.5, spike_grad=surrogate.atan())

        self.conv2 = nn.Conv2d(12, 64, 5)
        self.max2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.1)
        self.lif2 = snn.Leaky(beta=0.5, spike_grad=surrogate.atan())

        self.fc1 = nn.Linear(64 * 13 * 13, 11)
        self.lif3 = snn.Leaky(beta=0.5, spike_grad=surrogate.atan())

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spk_rec = []
        mem_rec = []

        num_steps = x.size(0)

        for step in range(num_steps):
            cur1 = self.max1(self.conv1(x[step]))
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.max2(self.conv2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc1(spk2.view(batch_size, -1))
            spk3, mem3 = self.lif3(cur3, mem3)

            if step == num_steps - 1:
                spk3 = self.apply_wta(spk3)

            spk_rec.append(spk3)
            mem_rec.append(mem3)

        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)

    @staticmethod
    def apply_wta(spike_output):
        spike_counts = spike_output.sum(dim=0)
        winner_neuron = torch.argmax(spike_counts)
        wta_output = torch.zeros_like(spike_output)
        wta_output[:, winner_neuron] = spike_output[:, winner_neuron]
        return wta_output

# Initialize model, optimizer, loss function, and scheduler
scnn = SCNN().to(device)
optimizer = torch.optim.Adam(scnn.parameters(), lr=2e-3, betas=(0.9, 0.999))
loss_fn = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

# Function to calculate accuracy
def calculate_accuracy_scnn(model, dataloader):
    correct = 0
    total = 0
    all_predicted = []
    all_labels = []

    model.eval()
    model.to(device)

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Accuracy Calculation", unit="batch", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            spike_counts = outputs.sum(dim=0)
            _, predicted = torch.max(spike_counts, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total if total > 0 else 0
    cm = confusion_matrix(all_labels, all_predicted)
    return accuracy, cm

# Calculate accuracy before training
scnn_accuracy_before, scnn_cm_before = calculate_accuracy_scnn(scnn, testloader)
print(f'Accuracy before training: {scnn_accuracy_before:.4f}%')
plot_confusion_matrix(scnn_cm_before, trainset)

# Initialize loss and accuracy history lists
loss_hist = []
acc_hist = []
# Training Loop
num_epochs = 1
for epoch in range(num_epochs):
    epoch_loss = 0
    epoch_accuracy = 0

    for i, (data, targets) in enumerate(tqdm(iter(trainloader), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch', leave=False)):
        data = data.to(device)
        targets = targets.to(device)

        scnn.train()
        spk_rec, _ = scnn(data)

        # Aggregate spikes over time dimension
        spk_rec = spk_rec.sum(dim=0)

        loss_val = loss_fn(spk_rec, targets)
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())
        epoch_loss += loss_val.item()

        # Calculate accuracy manually
        _, predicted = torch.max(spk_rec, 1)
        correct = (predicted == targets).sum().item()
        acc = correct / targets.size(0)
        acc_hist.append(acc)
        epoch_accuracy += acc

        # Clear CUDA cache after each batch
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    scheduler.step()
    epoch_loss /= len(trainloader)
    epoch_accuracy /= len(trainloader)
    print(f"Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}")


# Plotting Loss Curve
plt.figure(facecolor="w")
plt.plot(loss_hist, color='blue', linewidth=2, label='Train Loss')
plt.title("Loss Curves", fontsize=16)
plt.xlabel("Iteration", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()

# Calculate accuracy after training
scnn_accuracy_after, scnn_cm_after = calculate_accuracy_scnn(scnn, testloader)
print(f'Accuracy after training: {scnn_accuracy_after:.4f}%')
plot_confusion_matrix(scnn_cm_after, trainset)
