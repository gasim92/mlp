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


# Function to show images
def imshow(img, title):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()

# Visualize some test images and their predicted labels
dataiter = iter(test_loader)
images, labels = next(dataiter)

# Show input images
imshow(torchvision.utils.make_grid(images[:4]), "Input Images")

# Predict and show the output
model.eval()
with torch.no_grad():
    outputs = model(images[:4])
    _, predicted = torch.max(outputs, 1)

# Display images with their predictions
for i in range(4):
    imshow(images[i], f'Predicted: {predicted[i].item()}')