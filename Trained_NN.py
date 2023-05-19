# Now that the ResNet model has been trained on the dataset, we can test it on an image to classify the image as Benign or Malignant.


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
from PIL import Image


# trying to bypass SSL verification
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# Define hyperparameters
num_classes = 2 # Number of classes for breast cancer classification


# Load the pre-trained ResNet model
model = torch.hub.load("pytorch/vision", "vgg16", weights="IMAGENET1K_V1")

# Modify the last fully connected layer for the specific classification task
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, num_classes)

# Define the transformation to be applied to the input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the saved model state_dict
model_path = 'breast_cancer_classification_model.pth'
if os.path.isfile(model_path):
    model.load_state_dict(torch.load(model_path))
    print("Saved model loaded")
else:
    print("No saved model found")

# Establish path to folder of images
folder_path = "/path/to/folder"

# Get a sorted list of file names in alphabetical order
file_names = sorted(os.listdir(folder_path))

# Set the model to evaluation mode
model.eval()

# Iterate over each image in the folder
for image_name in file_names:
    # Path to the image
    image_path = os.path.join(folder_path, image_name)

    # Check if the file is a valid image file
    if not os.path.isfile(image_path) or not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff')):
        continue

    # Open the image file
    image = Image.open(image_path)

    # Convert the image to the 'RGB' mode
    image = image.convert('RGB')

    # Apply the transformation to the image
    input_image = transform(image)

    # Reshape the image tensor (optional, depends on the model's input requirements)
    input_image = input_image.unsqueeze(0)

    # Checking image tensor values 
    print("Input image tensor:", input_image)


    # Perform the forward pass
    with torch.no_grad():
        outputs = model(input_image)

    # Checking image tensor values
    print("Output tensor:", outputs)

    # Get the predicted class probabilities
    probs = torch.softmax(outputs, dim=1)
    predicted_class = torch.argmax(probs, dim=1)

    # Map the predicted class index to the corresponding label
    class_labels = ['Benign', 'Malignant']
    predicted_label = class_labels[predicted_class.item()]

    # Print the image name and its class label
    print("Image:", image_name)
    print("Predicted label:", predicted_label)
    print("Class probabilities:", probs.squeeze().tolist())
    print()
