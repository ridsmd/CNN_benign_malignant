# In summary, this code trains a VGG model from PyTorch with pre-trained weights. 
# It takes a parent folder with classification folders inside (Benign & Malignant)
# and puts 80% of the images in the training dataset and 20% in the validation dataset. 
# It trains itself on the training dataset, then tests itself on the validation dataset.

import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import vgg16
import random

# used this to bypass SSL verification (an error that was holding me back)
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# Define hyperparameters
num_classes = 2 # Number of classes for breast cancer classification
batch_size = 64
learning_rate = 0.001
num_epochs = 10

# Load the pre-trained ResNet model
# If we wanted to use a different CNN architecture, vgg16 with resnet50, inception_v3, alexnet, etc.
# If we switch the CNN used, last fully connected layer also needs to be modified.
# Depending on CNN used, different changes would need to be made.
model = torch.hub.load("pytorch/vision", "vgg16", weights="IMAGENET1K_V1")

# Modify the last fully connected layer for the specific classification task
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, num_classes)

# Define the transformation to be applied to the input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to match the input size of the network
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

# Load and preprocess the dataset. Include the quotations. the dataset is grabbed from a parent folder.
# Inside the parent folder, there should be 2 subfolders titled "Benign" and "Malignant" or whatever you want your labels to be.
dataset_path = "/path/to/dataset/parent_folder"

# Check if the dataset folder exists
if os.path.exists(dataset_path):
    print("Dataset folder exists")
else:
    print("Dataset folder not found")

# Load and preprocess the dataset
dataset = ImageFolder(root=dataset_path, transform=transform)

# Shuffle the dataset
random.shuffle(dataset.samples)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# the dataset I provide will be split into 80% training data and 20% testing / validation data.

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)



# Set the model to training mode
model.train()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    correct_predictions = 0

    for images, labels in train_loader:
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Track training loss and accuracy
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct_predictions += (predicted == labels).sum().item()

    # Calculate average loss and accuracy for the epoch
    epoch_loss = running_loss / len(train_dataset)
    epoch_accuracy = correct_predictions / len(train_dataset)

# at this point, the model should be finished using the training data. 
# Now, the validation dataset will be used to see how good the trained model is

     # Evaluate on the validation set
    model.eval()
    val_loss = 0.0
    val_correct_predictions = 0

    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_outputs = model(val_images)
            val_loss += criterion(val_outputs, val_labels).item() * val_images.size(0)
            _, val_predicted = torch.max(val_outputs.data, 1)
            val_correct_predictions += (val_predicted == val_labels).sum().item()

    val_epoch_loss = val_loss / len(val_dataset)
    val_epoch_accuracy = val_correct_predictions / len(val_dataset)

    # Print the loss and accuracy for each epoch
    print(f"Epoch [{epoch+1}/{num_epochs}] - "
          f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f} - "
          f"Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_epoch_accuracy:.4f}")

# Save the trained model
# I could have input the following 2 lines to see where the file will be saved (shows current directory). Otherwise, I can provide an absolute path to make it share in a specific location.
#current_dir = os.getcwd()
#print("Current working directory:", current_dir)
torch.save(model.state_dict(), 'breast_cancer_classification_model.pth')

# finally, model should be done with both training and testing, and model should be saved.