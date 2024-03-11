from tqdm import tqdm
from torchvision import transforms

# one epoch
def train_epoch(model, dataloader, loss_function, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    loss_values = []  # To store loss values for each batch
    
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Training')
    
    for i, (inputs, labels) in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = loss_function(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loss_values.append(loss.item())

        progress_bar.set_postfix(loss=loss.item())
        
    average_loss = sum(loss_values) / len(loss_values)
    print(f"Epoch {epoch} finished, average loss: {average_loss:.4f}")
    
    return running_loss