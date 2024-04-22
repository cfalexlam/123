# %% [markdown]
# ## Section 5.2: Using SEResNet
# 
# In this notebook, we are going to demonstrate using SEResNet to classify the 17 Category Flower Dataset.
# 
# #### i. Configure root path
# Configure the correct root directory for the dataset folder.

# %%
dataset_dir = "./dataset/"

import os
print(os.getcwd())

# %%
!pip install timm

# %% [markdown]
# #### 1. Load the model architecture
# For this demonstration, we will import the model architecutre from the timm library.

# %%
import timm

# %% [markdown]
# #### 2. Define training hyperparameters

# %%
learning_rates = [0.0001]
batch_sizes = [16] 
num_epochs = [40]

# %% [markdown]
# #### 3. Data transformation
# Define the dataloader and transforming function for the datasets

# %%
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# define dataset class
class FlowerDataset(ImageFolder):
    
    # Instanitiate dataloader
    def __init__(self, root_dir):
        
        # Define the transforming function
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # SEResNet uses 224x224 input
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Call parent constructor
        super().__init__(root=root_dir, transform=transform)
        
    # Get data loader
    def get_data_loader(self, batch_size, shuffle=False):
        return DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle)

# %% [markdown]
# #### 4. Training
# Define the train and evaluate function for each model

# %%
import torch
from torch.nn import CrossEntropyLoss, Linear
from torch.optim import Adam
from datetime import datetime

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"You are using: {device}")

def train(learning_rate, batch_size, epochs):
    
    # Print debug log message
    print(f"[{datetime.now()}] Training SEResNet-50 model (learning_rate={learning_rate}, batch_size={batch_size}, epochs={epochs})")
    
    # Print memory summary
    print(f"[{datetime.now()}] {torch.cuda.memory_summary()}")
    
    # Instaniate dataset objects
    train_dataset = FlowerDataset(dataset_dir+'/flowers/train')
    test_dataset = FlowerDataset(dataset_dir+'/flowers/test')
    val_dataset = FlowerDataset(dataset_dir+'/flowers/val')
    
    # Get dataloaders for the datasets
    train_loader = train_dataset.get_data_loader(batch_size, shuffle=True)
    test_loader = test_dataset.get_data_loader(batch_size)
    val_loader = val_dataset.get_data_loader(batch_size)
    
    # Create model
    model = timm.create_model('seresnet50', pretrained=False, num_classes=17).to(device)
    
    # Define criterion and optimizer
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # Create a dictonary to record model performance
    performance = { "train": [], "test": [], "val": []}
    
    # Evaluate accuracy for the model
    def evaluate(model, data_loader):
        model.eval()  # Make sure model is in evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in data_loader:
                outputs = model(images.to(device))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(device)).sum().item()
        return 100 * correct / total
    
    # Start training and evaluating
    for epoch in range(epochs):
        
        # Print debug log message
        print(f"[{datetime.now()}] Current epoch: {epoch + 1}")
        
        model.train()
        for inputs, labels in train_loader:
            images = inputs.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            del outputs, loss, images
            torch.cuda.empty_cache()
        
        model.eval()
        performance['train'].append(evaluate(model, train_loader))
        performance['test'].append(evaluate(model, test_loader))
        performance['val'].append(evaluate(model, val_loader))

        
    # Return result
    return performance

# %% [markdown]
# #### 5. Train multiple models
# Train multiple models iteratively with different defined hyperparameters

# %%
# Use to store result
performances = dict()

def train_models(learning_rates, batch_sizes, num_epochs):
    # Evaluating models with respect to different hyperparameters
    for learning_rate in learning_rates:
        for batch_size in batch_sizes:
            for epochs in num_epochs:
                performances[(learning_rate, batch_size, epochs)] = train(learning_rate, batch_size, epochs)
                torch.cuda.empty_cache()
                
train_models(learning_rates, batch_sizes, num_epochs)

# %% [markdown]
# #### 6. Model Evaluations
# First, we plot training, testing, and validation accuracies against number of epochs for individual models.

# %%
import numpy as np
from matplotlib import pyplot as plt

for model_hyperparams in performances.keys():
    
    # Destructure model hyperparameters from dictonary keys
    learning_rate, batch_size, epochs = model_hyperparams
    
    # Plot accuracies for different datasets
    for label in ["train", "test", "val"]:
        plt.plot([x + 1 for x in range(epochs)], performances[model_hyperparams][label], label=label)
    
    # Show plot
    plt.title(f"SEResNet Top-1 Accuracies on 17 Flowers Dataset (learning rate={learning_rate}, batch size={batch_size})")
    plt.xlabel("Number of Epoch")
    plt.ylabel("Top-1 Accuracy")
    plt.legend()
    plt.show()
    

# %% [markdown]
# Then, we plot testing accuracies against number of epochs for models with the same learning rate.

# %%
for target_learning_rate in learning_rates:
    
    for model_hyperparams in performances.keys():
    
        # Destructure model hyperparameters from dictonary keys
        learning_rate, batch_size, epochs = model_hyperparams
        
        if (learning_rate != target_learning_rate):
            continue
        
        plt.plot([x + 1 for x in range(epochs)], performances[model_hyperparams]["test"], label=f"batch_size={batch_size}")
        
    # Show plot
    plt.title(f"SEResNet Top-1 Accuracies on 17 Flowers (Test) Dataset (learning rate={target_learning_rate})")
    plt.xlabel("Number of Epoch")
    plt.ylabel("Top-1 Accuracy")
    plt.legend()
    plt.show()
    
    

# %% [markdown]
# Finally, we plot testing accuracies against number of epochs for models with the same batch size.

# %%
for target_batch_size in batch_sizes:
    
    for model_hyperparams in performances.keys():
    
        # Destructure model hyperparameters from dictonary keys
        learning_rate, batch_size, epochs = model_hyperparams
        
        if (batch_size != target_batch_size):
            continue
        
        plt.plot([x + 1 for x in range(epochs)], performances[model_hyperparams]["test"], label=f"learning_rate={learning_rate}")
        
    # Show plot
    plt.title(f"SEResNet Top-1 Accuracies on 17 Flowers (Test) Dataset (batch_size={batch_size})")
    plt.xlabel("Number of Epoch")
    plt.ylabel("Top-1 Accuracy")
    plt.legend()
    plt.show()

# %%



