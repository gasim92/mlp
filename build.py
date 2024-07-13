import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision
from PIL import Image


DEBUG = True
VISUALIZE=False

# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128) #fc- fully connected
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Initialize the model, loss function, and optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Lists to store the metrics
train_losses = []
val_accuracies = []

if not DEBUG:
    # Training loop
    num_epochs = 11
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)    
        
        print(f'Epoch [{epoch+1}/2], Loss: {loss.item():.4f}')
        
        
        # Validation loop
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            accuracy = 100 * correct / total
            val_accuracies.append(accuracy)
            print(f'Validation Accuracy: {accuracy:.2f}%')

    print("Training complete!")

    #validation loop
    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), "model/mnist_cnn.pt")

if VISUALIZE:    
    # Visualization
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 5))

    # Plotting training loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, 'r', label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.show()



if DEBUG:
    # Load the trained model state_dict
    model.load_state_dict(torch.load('model/mnist_cnn.pt'))
    #model.eval()

    # Define a transform to preprocess the image
    transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Resize to 28x28 pixels
    transforms.Grayscale(),       # Convert to grayscale (if not already)
    transforms.ToTensor(),        # Convert PIL Image to tensor
    ])

    # Load your own image
    img = Image.open('user/check1.png')
    print(img.size)

    try:

        # Preprocess the image
        img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    except Exception as e:
        print(f"Error occurred while processing image: {str(e)}")
        
    
    # Make prediction
    with torch.no_grad():
        output = model(img_tensor)

    # Get predicted class
    _, predicted = torch.max(output, 1)

    # Print the predicted class
    print(f'Predicted Class: {predicted.item()}')