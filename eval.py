import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from models.deep_vit import ViT

# Define the device
device = torch.device('mps')

# Load the saved model checkpoint
checkpoint_path = './artifacts/simplified_block_vit_epoch_2.pth'  # Update with your checkpoint path
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

def viz():
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
viz()