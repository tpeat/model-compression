import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from models.vit import ViT

# loss required to load models that use it
from loss.distillation_loss import DistillationLoss

# Define the device
device = torch.device('mps')

# Load the saved model checkpoint
checkpoint_path = './artifacts/student_vit_simplified_epoch_15.pth'  # Update with your checkpoint path
checkpoint = torch.load(checkpoint_path, map_location=device)

model = ViT(
    image_size=checkpoint['image_size'],
    patch_size=checkpoint['patch_size'],
    dim=checkpoint['dim'],
    depth=checkpoint['depth'],
    heads=checkpoint['heads'],
    mlp_dim=checkpoint['mlp_dim'],
    num_classes=checkpoint['num_classes']
)

model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

print(f"Model Parameters: {sum(p.numel() for p in model.parameters())}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

true_labels = []
pred_labels = []

with torch.no_grad():
    for data in tqdm(testloader, desc='Processing'):
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)
        
        true_labels += labels.cpu().tolist()
        pred_labels += predicted.cpu().tolist()


accuracy = accuracy_score(true_labels, pred_labels)

f1 = f1_score(true_labels, pred_labels, average='macro')  # Use 'micro' for micro-average F1 score

print(f'Accuracy of the model on the CIFAR-10 test images: {accuracy * 100:.2f}%')
print(f'F1 Score of the model on the CIFAR-10 test images: {f1:.2f}')


# viz some predictions
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def imshow(img):
    """Function to show an image"""
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')

def viz_predictions():
    dataiter = iter(testloader)
    images, labels = [], []

    # only asked for four samples, could change later
    for _ in range(4):
        image, label = next(dataiter)
        images.append(image)
        labels.append(label)

    images = torch.cat(images).to(device)
    labels = torch.cat(labels)

    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    fig = plt.figure(figsize=(10, 10))

    for i in range(4):
        ax = fig.add_subplot(2, 2, i+1, xticks=[], yticks=[])
        imshow(images.cpu()[i])
        ax.set_title(f"True: {class_names[labels[i]]}, Predicted: {class_names[predicted[i]]}")

    plt.show()

# call visualization
viz_predictions()

# visualize training times on linear plot
def read_training_times(file_path):
    with open(file_path, 'r') as file:
        times = file.readlines()
        times = [float(time.strip()) for time in times]
    return times

def plot_training_times(file1, file2):
    times1 = read_training_times(file1)
    times2 = read_training_times(file2)

    min_epochs = min(len(times1), len(times2))

    epochs1 = range(1, len(times1) + 1)
    epochs2 = range(1, len(times2) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs1, times1, label=file1, marker='o')
    plt.plot(epochs2, times2, label=file2, marker='x')

    plt.xlabel('Epoch')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
file1 = './artifacts/simplified_block_vit_training_times.txt'
file2 = './artifacts/student_vit_training_times.txt'
plot_training_times(file1, file2)