import os
import torch
import torch.nn as nn #Step 2
import torch.nn.functional as F
import torch.optim as optim #Step 4
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image
from torchinfo import summary

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
#Resnet Architecture
class myResNet20(nn.Module):
    def __init__(self,num_classes=2):
        super(myResNet20,self).__init__()
        #Initial
        self.conv1=nn.Conv2d(3,16,kernel_size=3,padding=1,
                             bias=False,stride=1)
        self.bn1=nn.BatchNorm2d(16)

        #Stage 1
        #Block 1
        self.s1_b1_conv1=nn.Conv2d(16,16,kernel_size=3,
                                    padding=1,bias=False,stride=1)
        self.s1_b1_bn1=nn.BatchNorm2d(16)

        self.s1_b1_conv2=nn.Conv2d(16,16,kernel_size=3,
                                    padding=1,bias=False,stride=1)
        self.s1_b1_bn2=nn.BatchNorm2d(16)

        #Block 2
        self.s1_b2_conv1=nn.Conv2d(16,16,kernel_size=3,
                                    padding=1,bias=False,stride=1)
        self.s1_b2_bn1=nn.BatchNorm2d(16)
        
        self.s1_b2_conv2=nn.Conv2d(16,16,kernel_size=3,
                                    padding=1,bias=False,stride=1)
        self.s1_b2_bn2=nn.BatchNorm2d(16)

        #Block 3
        self.s1_b3_conv1=nn.Conv2d(16,16,kernel_size=3,
                                    padding=1,bias=False,stride=1)
        self.s1_b3_bn1=nn.BatchNorm2d(16)
                
        self.s1_b3_conv2=nn.Conv2d(16,16,kernel_size=3,
                                    padding=1,bias=False,stride=1)
        self.s1_b3_bn2=nn.BatchNorm2d(16)

        #Stage 2
        #Block 1
        self.s2_b1_conv1=nn.Conv2d(16,32,kernel_size=3,
                                    padding=1,bias=False,stride=2)
        self.s2_b1_bn1=nn.BatchNorm2d(32)
        
        self.s2_b1_conv2=nn.Conv2d(32,32,kernel_size=3,
                                    padding=1,bias=False,stride=1)
        self.s2_b1_bn2=nn.BatchNorm2d(32)

        self.skip1=nn.Conv2d(16,32,kernel_size=1,
                             padding=0,bias=False,stride=2)
        
        #Block 2
        self.s2_b2_conv1=nn.Conv2d(32,32,kernel_size=3,
                                    padding=1,bias=False,stride=1)
        self.s2_b2_bn1=nn.BatchNorm2d(32)
                
        self.s2_b2_conv2=nn.Conv2d(32,32,kernel_size=3,
                                    padding=1,bias=False,stride=1)
        self.s2_b2_bn2=nn.BatchNorm2d(32)
        
        #Block 3
        self.s2_b3_conv1=nn.Conv2d(32,32,kernel_size=3,
                                    padding=1,bias=False,stride=1)
        self.s2_b3_bn1=nn.BatchNorm2d(32)
                        
        self.s2_b3_conv2=nn.Conv2d(32,32,kernel_size=3,
                                    padding=1,bias=False,stride=1)
        self.s2_b3_bn2=nn.BatchNorm2d(32)

        #Stage 3
        #Block 1
        self.s3_b1_conv1=nn.Conv2d(32,64,kernel_size=3,
                                    padding=1,bias=False,stride=2)
        self.s3_b1_bn1=nn.BatchNorm2d(64)
                
        self.s3_b1_conv2=nn.Conv2d(64,64,kernel_size=3,
                                    padding=1,bias=False,stride=1)
        self.s3_b1_bn2=nn.BatchNorm2d(64)
        
        self.skip2=nn.Conv2d(32,64,kernel_size=1,
                                    padding=0,bias=False,stride=2)
                
        #Block 2
        self.s3_b2_conv1=nn.Conv2d(64,64,kernel_size=3,
                                    padding=1,bias=False,stride=1)
        self.s3_b2_bn1=nn.BatchNorm2d(64)
                        
        self.s3_b2_conv2=nn.Conv2d(64,64,kernel_size=3,
                                    padding=1,bias=False,stride=1)
        self.s3_b2_bn2=nn.BatchNorm2d(64)
                
        #Block 3
        self.s3_b3_conv1=nn.Conv2d(64,64,kernel_size=3,
                                    padding=1,bias=False,stride=1)
        self.s3_b3_bn1=nn.BatchNorm2d(64)
                                
        self.s3_b3_conv2=nn.Conv2d(64,64,kernel_size=3,
                                    padding=1,bias=False,stride=1)
        self.s3_b3_bn2=nn.BatchNorm2d(64)

        #Classifier layer
        self.linear = nn.Linear(64, num_classes)
    def forward(self, x):
            # Initial Layer
            out = F.relu(self.bn1(self.conv1(x)))
            
            # --- STAGE 1 ---
            # Block 1.1
            identity = out
            out = F.relu(self.s1_b1_bn1(self.s1_b1_conv1(out)))
            out = self.s1_b1_bn2(self.s1_b1_conv2(out))
            out += identity  # Skip connection
            out = F.relu(out)
    
            # Block 1.2
            identity = out
            out = F.relu(self.s1_b2_bn1(self.s1_b2_conv1(out)))
            out = self.s1_b2_bn2(self.s1_b2_conv2(out))
            out += identity
            out = F.relu(out)
    
            # Block 1.3
            identity = out
            out = F.relu(self.s1_b3_bn1(self.s1_b3_conv1(out)))
            out = self.s1_b3_bn2(self.s1_b3_conv2(out))
            out += identity
            out = F.relu(out)
    
            # --- STAGE 2 ---
    
            identity = self.skip1(out)
            out = F.relu(self.s2_b1_bn1(self.s2_b1_conv1(out)))
            out = self.s2_b1_bn2(self.s2_b1_conv2(out))
            out += identity
            out = F.relu(out)
    
            # Block 2.2
            identity = out
            out = F.relu(self.s2_b2_bn1(self.s2_b2_conv1(out)))
            out = self.s2_b2_bn2(self.s2_b2_conv2(out))
            out += identity
            out = F.relu(out)
    
            # Block 2.3
            identity = out
            out = F.relu(self.s2_b3_bn1(self.s2_b3_conv1(out)))
            out = self.s2_b3_bn2(self.s2_b3_conv2(out))
            out += identity
            out = F.relu(out)
    
            # --- STAGE 3 ---
            identity = self.skip2(out)
            out = F.relu(self.s3_b1_bn1(self.s3_b1_conv1(out)))
            out = self.s3_b1_bn2(self.s3_b1_conv2(out))
            out += identity
            out = F.relu(out)
    
            # Block 3.2
            identity = out
            out = F.relu(self.s3_b2_bn1(self.s3_b2_conv1(out)))
            out = self.s3_b2_bn2(self.s3_b2_conv2(out))
            out += identity
            out = F.relu(out)
    
            # Block 3.3
            identity = out
            out = F.relu(self.s3_b3_bn1(self.s3_b3_conv1(out)))
            out = self.s3_b3_bn2(self.s3_b3_conv2(out))
            out += identity
            out = F.relu(out)
    
            
            out = F.avg_pool2d(out, out.size()[3]) 
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            
            return out
    
#Hyperparameter setup
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate=0.001
num_epochs=5
    
#Loss function and Optimizer
model=myResNet20().to(device)
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
print(summary(model))
