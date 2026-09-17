import os
import torch
import torch.nn as nn #Step 2
import torch.optim as optim #Step 4
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image

root="./images/PetImages"
def is_valid_image(path):
    try:
        img = Image.open(path)
        img.verify() # Checks if the file is a valid image without loading it fully
        return True
    except:
        return False

batch_size=64
transform=transforms.Compose([transforms.Resize((32,32)),
   transforms.ToTensor(),#pixel is in [0,1]
   transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])

full_dataset=datasets.ImageFolder(root=root,transform=transform,
                                  is_valid_file=is_valid_image)

total_size=len(full_dataset)
train_size=int(0.8*total_size)
val_size=total_size-train_size
#val_size=0.2*total_size
train_dataset,val_dataset=random_split(full_dataset,
                                       [train_size,val_size],
                                       generator=torch.Generator().manual_seed(32))

train_loader=DataLoader(train_dataset,
                        batch_size=64,
                        shuffle=True)
val_loader=DataLoader(val_dataset,
                        batch_size=64,
                        shuffle=False)

#Define Model
#LeNet Architecture
class LeNet5(nn.Module):
   def __init__(self):
      super(LeNet5,self).__init__()
      self.conv_layer=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5,stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2,stride=2))
      self.fc_layer=nn.Sequential(
            nn.Linear(in_features=400,out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120,out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84,out_features=10))

   def forward(self,x):
      x=self.conv_layer(x)
      x=x.view(-1,16*5*5)#Flatten the tensor
      x=self.fc_layer(x)
      return(x)

#Hyperparameter setup
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate=0.001
num_epochs=5

#Loss function and Optimizer
model=LeNet5().to(device)
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=learning_rate)

model.train()
#Training Loop
print(f"Training on {device}")
for epoch in range(num_epochs):
   loss=0
   for i,(images,labels) in enumerate(train_loader):
      images,labels=images.to(device), labels.to(device)
      #Forward pass
      output=model(images)
      loss=criterion(output,labels)
      #Backpropagation
      optimizer.zero_grad()
      loss.backward()#Evaluate gradient wrt loss
      optimizer.step()#Update weights
      loss=loss+loss.item()
   print(f"Epoch:{epoch+1},loss:{loss/len(train_loader)}")

#Evaluation
model.eval()
correct=0
total=0

with torch.no_grad():
   for images,labels in val_loader:
      images, labels = images.to(device), labels.to(device)
      output=model(images)
      _,predicted=torch.max(output.data,1)
      total=total+labels.size(0)
      correct=correct+(predicted==labels).sum().item()
print(f"Test Accuracy:{correct/total}")

