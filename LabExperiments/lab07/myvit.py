import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#HyperParameter setup
BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.05
IMAGE_SIZE = 32
PATCH_SIZE = 4
IN_CHANNELS = 3
NUM_CLASSES = 10

# ViT Architecture Hyperparameters (Tiny-ViT)
EMBED_DIM = 128
DEPTH = 4
NUM_HEADS = 4
MLP_RATIO = 2.0
DROPOUT = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

#Generate Train Loaders
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = datasets.ImageFolder(root="data/cifar10/cifar10/train",transform=transform_train)
train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

testset = datasets.ImageFolder(root="data/cifar10/cifar10/test",transform=transform_test)
train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


#Patch Embedding Class
class PatchEmbedding(nn.Module):

  def __init__(self, in_channels, patch_size, emb_size, img_size, num_classes):
    super(PatchEmbedding,self).__init__()
    self.patch_size = patch_size
    self.num_patches = (img_size // patch_size) ** 2

    # Convolutional projection maps image patches to embedding space efficiently
    self.projection = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    # Learnable [CLS] token and Position Embeddings
    self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
    self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, emb_size))
    self.dropout = nn.Dropout(DROPOUT)

  def forward(self, x):
    b, c, h, w = x.shape
    x = self.projection(x)  # (B, Emb_Size, H/P, W/P)
    x = x.flatten(2).transpose(1, 2)  # (B, Num_Patches, Emb_Size)

    # Prepend [CLS] token
    cls_tokens = self.cls_token.expand(b, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)  # (B, Num_Patches + 1, Emb_Size)

    # Add position embeddings
    x = x + self.pos_embedding
    return self.dropout(x)


#ViT Architecture
class TinyViT(nn.Module):

  def __init__(
      self,
      img_size=IMAGE_SIZE,
      patch_size=PATCH_SIZE,
      in_channels=IN_CHANNELS,
      num_classes=NUM_CLASSES,
      emb_size=EMBED_DIM,
      depth=DEPTH,
      num_heads=NUM_HEADS,
      mlp_ratio=MLP_RATIO,
  ):
    super(TinyViT,self).__init__()

    self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_size, img_size, num_classes)

    # Transformer Encoder Blocks using PyTorch's built-in components
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=emb_size,
        nhead=num_heads,
        dim_feedforward=int(emb_size * mlp_ratio),
        dropout=DROPOUT,
        activation="gelu",
        batch_first=True,
    )
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    # Classification Head
    self.norm = nn.LayerNorm(emb_size)
    self.head = nn.Linear(emb_size, num_classes)

  def forward(self, x):
    x = self.patch_embed(x)
    x = self.transformer_encoder(x)

    # Extract the [CLS] token output for classification (index 0)
    cls_output = x[:, 0]
    cls_output = self.norm(cls_output)
    out = self.head(cls_output)
    return out


# Instantiate model, loss, and optimizer
model = TinyViT().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW( model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

#Training
for epoch in range(EPOCHS):
  model.train()
  running_loss = 0.0
  correct = 0
  total = 0

  for images, labels in train_loader:
    images, labels = images.to(DEVICE), labels.to(DEVICE)

    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    _, predicted = outputs.max(1)
    total += labels.size(0)
    correct += (predicted==labels).sum().item()

  scheduler.step()
  train_acc = correct / total
  train_loss = running_loss / len(train_loader)
  print(f"Epoch {epoch}: Loss:{train_loss};Accuracy:{train_acc}")

# Evaluation on Test Set
model.eval()
test_loss = 0.0
correct_test = 0
total_test = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total_test += labels.size(0)
    correct_test += (predicted==labels).sum().item()

    test_acc = correct_test / total_test
print(f"Test Accuracy:{test_acc}")

