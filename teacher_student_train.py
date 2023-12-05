import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import os
from models.vit import ViT
from models.teacher_vit import ViT as TeacherViT
from loss.distillation_loss import DistillationLoss

# load datasets
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

device = torch.device('mps')

def load_teacher_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    teacher_model = TeacherViT(
        image_size=checkpoint['image_size'],
        patch_size=checkpoint['patch_size'],
        dim=checkpoint['dim'],
        depth=checkpoint['depth'],
        heads=checkpoint['heads'],
        mlp_dim=checkpoint['mlp_dim'],
        num_classes=checkpoint['num_classes']
    )
    teacher_model.load_state_dict(checkpoint['model_state_dict'])
    teacher_model.to(device)
    teacher_model.eval()  # Set the model to evaluation mode
    return teacher_model


teacher_model_path = './artifacts/simplified_block_vit_epoch_15.pth'
teacher_model = load_teacher_model(teacher_model_path, device)

# get model, also add option to resume from checkpoint by loading these things:
image_size = 32  # CIFAR-10 images are 32x32
patch_size = 4   # Size of the patches to be extracted from the images
dim = 256        # Dimension of the transformer layers
depth = 1        # Number of transformer blocks
heads = 4        # Number of heads for the multi-head attention
mlp_dim = 512    # Dimension of the feed-forward network
num_classes = 10

# will be student model
model = ViT(
    image_size=image_size,
    patch_size=patch_size,
    dim=dim,
    depth=depth,
    heads=heads,
    mlp_dim=mlp_dim,
    num_classes=num_classes
)

print(f"Model Parameters: {sum(p.numel() for p in model.parameters())}")


# one epoch
def train_epoch(student_model, teacher_model, dataloader, loss_function, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    loss_values = []  # To store loss values for each batch
    
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Training')
    
    for i, (inputs, labels) in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Get outputs from the student model
        student_outputs = student_model(inputs)

        # Get outputs from the teacher model
        with torch.no_grad():
            teacher_outputs = teacher_model(inputs)

        # Calculate distillation loss
        loss = loss_function(student_outputs, teacher_outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loss_values.append(loss.item())

        progress_bar.set_postfix(loss=loss.item())
        
    average_loss = sum(loss_values) / len(loss_values)
    print(f"Epoch {epoch} finished, average loss: {average_loss:.4f}")
    
    return running_loss



model.to(device)

# model name
model_name = 'student_vit'

num_epochs = 25
checkpoint_freq = 5
artifact_directory = './artifacts'
os.makedirs(artifact_directory, exist_ok=True)



# Use DistillationLoss as your loss function
loss_function = DistillationLoss(alpha=0.5, temperature=4)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

# log training time
epoch_times = []

for epoch in range(num_epochs):
    # start timer
    start_time = time.time()

    train_epoch(model, teacher_model, train_loader, loss_function, optimizer, device, epoch)
    
    end_time = time.time()
    duration = end_time - start_time
    epoch_times.append(duration)

    # save model
    if (epoch + 1) % checkpoint_freq == 0:
        checkpoint_path = os.path.join(artifact_directory, f'{model_name}_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_function,
            'image_size': image_size,
            'patch_size': patch_size,
            'dim': dim,
            'depth': depth,
            'heads': heads,
            'mlp_dim': mlp_dim,
            'num_classes': num_classes
        }, checkpoint_path)
        print(f'Checkpoint saved to {checkpoint_path}')

        # also log training times so far, for later use
        training_times_path = os.path.join(artifact_directory, f'{model_name}_training_times.txt')
        with open(training_times_path, 'w') as file:
            for t in epoch_times:
                file.write(f"{t}\n")
