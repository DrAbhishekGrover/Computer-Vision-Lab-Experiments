import os
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F


class LFWDataset(Dataset):
    def __init__(self,image_dir,match_csv,mismatch_csv,mytransform=None):
        self.image_dir=image_dir
        self.transform=mytransform
        match=pd.read_csv(match_csv)
        mismatch=pd.read_csv(mismatch_csv)
        self.pairs=[]
        for _,row in match.iterrows():
            name=row['name']
            img1=int(row['imagenum1'])
            img2=int(row['imagenum2'])
            path1=os.path.join(image_dir,"lfw-deepfunneled",
                               "lfw-deepfunneled",name,
                               f"{name}_{img1:04d}.jpg",)
            path2=os.path.join(image_dir,"lfw-deepfunneled",
                               "lfw-deepfunneled",name,
                               f"{name}_{img2:04d}.jpg",)
            self.pairs.append((path1,path2,0))
        #For Mismatch Pairs
        for _,row in mismatch.iterrows():
            name1=row['name1']
            name2=row['name2']
            img1=int(row['imagenum1'])
            img2=int(row['imagenum2'])
            path1=os.path.join(image_dir,"lfw-deepfunneled",
                                "lfw-deepfunneled",name1,
                                f"{name1}_{img1:04d}.jpg",)
            path2=os.path.join(image_dir,"lfw-deepfunneled",
                                "lfw-deepfunneled",name2,
                                f"{name2}_{img2:04d}.jpg",)
            self.pairs.append((path1,path2,1))

    def __len__(self):
        return(len(self.pairs))
    def __getitem__(self, idx):
        path1,path2,label=self.pairs[idx]
        img1=Image.open(path1).convert("RGB")
        img2=Image.open(path2).convert("RGB")
        if self.transform:
            image1=self.transform(img1)
            image2=self.transform(img2)
        label=torch.tensor(label,dtype=torch.float32)
        return(image1,image2,label)

transform=transforms.Compose([transforms.Resize((100,100)),
                             transforms.ToTensor(),
                             transforms.Normalize(
                                mean=[0.485,0.456,0.406],
                                std=[0.229,0.224,0.225] 
                             )])

data_dir="images/lfw"
train_dataset=LFWDataset(data_dir,
                         os.path.join(data_dir,"matchpairsDevTrain.csv"),
                         os.path.join(data_dir,"mismatchpairsDevTrain.csv"),
                         mytransform=transform)
train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)

test_dataset=LFWDataset(data_dir,
                         os.path.join(data_dir,"matchpairsDevTest.csv"),
                         os.path.join(data_dir,"mismatchpairsDevTest.csv"),
                         mytransform=transform)
test_loader=DataLoader(test_dataset,batch_size=32,shuffle=False)

#img1,img2,label=next(iter(train_loader))
#print(img1.shape)
#print(img2.shape)
#print(label.shape)
#print(label)

#Class LeNet
class myLeNet(nn.Module):
    def __init__(self,embedding_dim=50):
        super(myLeNet,self).__init__()
        #input:3x100x100
        self.conv1=nn.Conv2d(3,6,kernel_size=5)#6x96x96
        self.pool1=nn.MaxPool2d(2,2)#6x48x48
        self.conv2=nn.Conv2d(6,16,kernel_size=5)#16x44x44
        self.pool2=nn.MaxPool2d(2,2)#16x22x22
        #Feedforward layer
        self.fc1=nn.Linear(16*22*22,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,embedding_dim)

    def forward(self,x):
        x=self.pool1(F.relu(self.conv1(x)))
        x=self.pool2(F.relu(self.conv2(x)))
        #16x22x22 to a vector (view)
        #x=[32x16x22x22]
        x=x.view(x.size(0),-1)
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.fc3(x)
        return(x)

class mySiamese(nn.Module):
    def __init__(self,backbone):
        super(mySiamese,self).__init__()
        self.backbone=backbone

    def forward(self,img1,img2):
        out1=self.backbone(img1)
        out2=self.backbone(img2)
        return(out1,out2)

class mycontrastiveloss(nn.Module):
    def __init__(self,margin=2.0):
        super(mycontrastiveloss,self).__init__()
        self.margin=margin
    def forward(self,out1,out2,label):
        d=F.pairwise_distance(out1,out2)
        loss=torch.mean((1-label)*torch.pow(d,2)+(label)*torch.pow(
            (torch.clamp(self.margin-d,min=0.0)),2))
        return(loss)

#Create objects
device=('cuda' if torch.cuda.is_available() else 'cpu')
net=myLeNet(embedding_dim=50)
siamese_net=mySiamese(net).to(device)
criterion=mycontrastiveloss(margin=2.0)
optimizer=torch.optim.Adam(siamese_net.parameters(),lr=0.0005)

#Training loop
siamese_net.train()
for epoch in range(5):
    for i,data in enumerate(train_loader):
        img0,img1,label=data
        img0,img1,label=img0.to(device),img1.to(device),label.to(device)
        out1,out2=siamese_net(img0,img1)
        loss=criterion(out1,out2,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}: Loss {loss}")

#Evaluation
threshold=1.0
correct=0
total=0

siamese_net.eval()
with torch.no_grad():
    for i,data in enumerate(test_loader):
        img0,img1,label=data
        img0,img1,label=img0.to(device),img1.to(device),label.to(device)
        out1,out2=siamese_net(img0,img1)
        d=F.pairwise_distance(out1,out2)
        prediction=(d>threshold).float()
        correct=correct+(prediction==label).sum().item()
        total=total+label.size(0)
print(f"Test Accuracy:{correct/total}")




    
        
