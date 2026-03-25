# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.
## Problem Statement and Dataset

The objective of this experiment is to perform image classification using transfer learning with a pre-trained deep learning model. A convolutional neural network based on VGG19 is used as the base model to classify images into two categories. Instead of training the network from scratch, the pre-trained weights learned from ImageNet are utilized to extract high-level features. The final fully connected layer of the model is modified to suit the binary classification task. The model is trained and evaluated using loss curves, confusion matrix, and classification report to measure performance.

<img width="732" height="163" alt="image" src="https://github.com/user-attachments/assets/2e85732d-4610-4572-9412-52e0e95eed57" />


## DESIGN STEPS

### STEP 1:

Import required libraries, extract the dataset, apply image transformations, and create DataLoader objects for training and testing.

### STEP 2:

Load the pre-trained VGG19 model in PyTorch, freeze feature layers, and modify the final layer for binary classification.

### STEP 3:

Define the loss function and optimizer, then train the model while computing training and validation loss for each epoch.

### STEP 4:

Evaluate the model on the test dataset and compute accuracy, confusion matrix, and classification report.

### STEP 5:

Perform prediction on individual test images and display the actual and predicted class labels.

## PROGRAM

```python
# Load Pretrained Model and Modify for Transfer Learning
from torchvision.models.vgg import VGG19_Weights
model=models.vgg19(weights=VGG19_Weights.DEFAULT)

# Modify the final fully connected layer to match the dataset classes
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features,1)

# Include the Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

# Train the model
def train_model(model, train_loader,test_loader,num_epochs=100):
  train_losses=[]
  val_losses=[]
  model.train()
  for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss=criterion(outputs,labels.unsqueeze(1).float())

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss/len(train_loader))


    # Compute validation loss
    model.eval()
    val_loss=0.0
    with torch.no_grad():
      for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss=criterion(outputs,labels.unsqueeze(1).float())
        val_loss+=loss.item()
    val_losses.append(val_loss/len(test_loader))
    model.train()

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

  # Plot training and validation loss - Moved outside the loop
  print("Name: Ahil Santo A")
  print("Register Number: 212224040018")
  plt.figure(figsize=(8, 6))
  plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
  plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.title('Training and Validation Loss')
  plt.legend()
  plt.show()

# Train the model
train_model(model, train_loader,test_loader)

```

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot

<img width="610" height="214" alt="image" src="https://github.com/user-attachments/assets/987c4682-782c-4a3d-9574-d213aa72a05a" />

<img width="659" height="212" alt="image" src="https://github.com/user-attachments/assets/67b0b335-dd84-47e9-9c57-1651637b8ccf" />

<img width="983" height="753" alt="image" src="https://github.com/user-attachments/assets/c411f4e4-24b6-470b-8f4e-860cabeb3354" />


### Confusion Matrix

<img width="941" height="717" alt="image" src="https://github.com/user-attachments/assets/ba56ff2b-e4aa-4996-b61d-f30a54a55e74" />

### Classification Report

<img width="790" height="340" alt="image" src="https://github.com/user-attachments/assets/b483923e-8e49-4088-939e-2b728ca957fa" />

### New Sample Prediction

<img width="693" height="528" alt="image" src="https://github.com/user-attachments/assets/5c195c02-2d3d-4bea-9cd3-185a6ee53027" />

<img width="820" height="640" alt="image" src="https://github.com/user-attachments/assets/36dc7e29-9743-416b-8a83-8241eb40f5d5" />

## RESULT

Thus to Implement Transfer Learning for classification using VGG-19 architecture is done successfully.
