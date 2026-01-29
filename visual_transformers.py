#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 15:28:38 2025

@author: Nick Sokolov
@author: Casimiro Barreto
"""

# System imports
import os
import random
from datetime import datetime
import json
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# TORCH imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# Verify if cuda is available
device = 'cuda' if torch.cuda.is_available() else "cpu"
print(f"Using {device} for tensor processing.")

# Randomize
if device == 'cuda':
    torch.cuda.manual_seed(96)
else:
    torch.manual_seed(96)

random.seed(69)

# Training parameters
BATCH_SIZE = 128
EPOCHS = 80
LEARNING_RATE = 3e-4
PATCH_SIZE = 4
NUM_CLASSES = 100
IMAGE_SIZE = 32
CHANNELS = 3
EMBED_DIM = 256
NUM_HEADS = 8
DEPTH = 6
MLP_DIM = 512
DROP_RATE = 0.10
WORK_DIR = "working/"
SAVE_DIR = "working/weights"
PLOT_DIR = "working/plots"

# Patch embedding
class PatchEmbedding(nn.Module):
    def __init__(self, img_size = 32, patch_size = 4, in_channels = 3, embed_dim=256):
        super().__init__()

        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.patch_size = patch_size
        self.num_patches = (img_size//patch_size)**2
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.conv(x)                 #(B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2) #(B, num_patches, embed_dim)
        return x
    
# Attention mechanism
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        assert dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.num_heads = num_heads

        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias = False)
        self.out = nn.Linear(dim, dim, bias = False)

        self.scale = 1.0 / (self.head_dim ** 0.5)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(self,  x, mask = None,  return_attn=False):
        B, num_patches, embed_dim = x.shape

        qkv = self.qkv(x) # (B, num_patches, 3*embed_dim)
        qkv = qkv.reshape(B, num_patches, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) #(3, B, num_heads, num_patches, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, num_heads, num_patches, head_dim)

                                        #How important it is for token i to pay attention to token j.
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale #[B, num_heads, N, N]

        if mask is not None:
            # mask: (B, 1, N, N) or (1, 1, N, N)
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_probs = attn_scores.softmax(dim=-1) #[B, num_heads, N, N]
        attn_probs = self.attn_dropout(attn_probs)
        attn_output = attn_probs @ v  # (B, num_heads, num_patches, head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(B, num_patches, embed_dim)

        if return_attn:
          return self.out(attn_output), attn_probs
        else:
          return self.out(attn_output) #(B, num_patches, embed_dim)

# The encoder
class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attn=False):
      if return_attn:
        attn_out, attn_weights = self.attn(self.norm1(x), return_attn=True)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x, attn_weights
      else:
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x
    
# The Vision Transformer
class VisualTransformer(nn.Module):
    def __init__(self,num_classes, img_size=32, patch_size=4, in_channels=3, embed_dim=256,
                 num_layers=6, num_heads=7, mlp_dim=512, dropout=0.1):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + self.patch_embed.num_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)

        self.encoder_blocks = nn.ModuleList([
                TransformerEncoderBlock(embed_dim, num_heads, mlp_dim, dropout)
                for _ in range(num_layers)
            ])

        self.norm = nn.LayerNorm(embed_dim)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(in_features=embed_dim, out_features=num_classes)
        )

    def forward(self, x, return_attn = False):
        B = x.size(0)
        x = self.patch_embed(x)  # (B, N, D)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 1+N, D)
        x = x + self.pos_embed
        x = self.dropout(x)

        attn_maps = []
        for block in self.encoder_blocks:
          if return_attn:
            x, attn = block(x, return_attn=True)
            attn_maps.append(attn)  # (B, heads, N, N)
          else:
            x = block(x) # (B, 1+N, D)

        x = self.norm(x)

        out = self.mlp_head(x[:, 0, :])

        if return_attn:
          return out, attn_maps
        else:
          return out

#
def train_epoch(model, loader, optimizer, criterion, progress_bar):
    model.train()

    total_loss, correct = 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()

        progress_bar.update(1)
        progress_bar.set_postfix(loss=loss.item())
    return total_loss/len(loader.dataset), correct/len(loader.dataset)

def evaluate_epoch(model, loader):
    model.eval()
    correct = 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
    return correct/len(loader.dataset)

def training_model(model, loader, test_loader, optimizer, criterion, scheduler, num_epochs, start_epoch=0):
    train_accurs, train_losses = [], []
    test_accurs = []

    for epoch in range(start_epoch, num_epochs):
        progress_bar = tqdm(loader, desc=f'Epoch {epoch+1}/{num_epochs}', dynamic_ncols=True)
        train_loss, train_acc = train_epoch(model, loader, optimizer, criterion, progress_bar)
        train_accurs.append(train_acc)
        train_losses.append(train_loss)

        test_acc = evaluate_epoch(model, test_loader)
        test_accurs.append(test_acc)

        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        print(f'\nEpoch [{epoch+1}/{num_epochs}] — LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}')

        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(SAVE_DIR, f"vit_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✅ The weights were saved: {checkpoint_path}")

    return train_losses, train_accurs, test_accurs

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trained parameters: {trainable:,}")

def denormalize(img_tensor, mean =(0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return img_tensor * std + mean

def main():
    #
    global BATCH_SIZE
    global EPOCHS
    global LEARNING_RATE
    global PATCH_SIZE
    global NUM_CLASSES
    global IMAGE_SIZE
    global CHANNELS
    global EMBED_DIM
    global NUM_HEADS
    global DEPTH
    global MLP_DIM
    global DROP_RATE
    global WORK_DIR
    global SAVE_DIR
    global PLOT_DIR
    
    #
    print(f"*** TRAINING STARTED AT: {datetime.now()}")
    print("*** PARAMETERS:")
    print(f"BATCH SIZE: {BATCH_SIZE}")
    print(f"EPOCHS: {EPOCHS}")
    print(f"LEARNING RATE: {LEARNING_RATE}")
    print(f"PATCH SIZE: {PATCH_SIZE}")
    print(f"NUM CLASSES: {NUM_CLASSES}")
    print(f"IMAGE SIZE: {IMAGE_SIZE}")
    print(f"CHANNELS: {CHANNELS}")
    print(f"EMBED DIM: {EMBED_DIM}")
    print(f"NUM HEADS: {NUM_HEADS}")
    print(f"DEPTH: {DEPTH}")
    print(f"MLP DIM: {MLP_DIM}")
    print(f"DROP RATE: {DROP_RATE}")
    print(f"WORK DIR: {WORK_DIR}")
    print(f"SAVE_DIR: {SAVE_DIR}")
    print(f"PLOT DIR: {PLOT_DIR}")
    
    train_transforms = transforms.Compose([
        transforms.Resize((70, 70)),
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'),
        ])
    
    #
    val_transforms = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    # loading test dataset
    start = datetime.now()
    print(f"Loading test set started at {start}")
    test_dataset = datasets.CIFAR100(root = WORK_DIR,
                                    train = False,
                                    download = True,
                                    transform = val_transforms)
    end = datetime.now()
    print(f"Test set loaded. Elapsed time: {end - start}")

    # loading train set
    start = datetime.now()
    print(f"Loading training set started at: {start}")
    train_dataset = datasets.CIFAR100(root = WORK_DIR,
                                     train = True,
                                     download = True,
                                     transform = train_transforms )
    end = datetime.now()
    print(f"Train set loaded. Elapsed time: {end - start}")
    
    plt.imshow(train_dataset[1054][0].permute(1, 2, 0))
    
    train_loader = DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = True)
    test_loader = DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, shuffle = False)
    
    model = VisualTransformer(NUM_CLASSES, img_size=IMAGE_SIZE, 
                              patch_size=PATCH_SIZE, in_channels=CHANNELS, 
                              embed_dim=EMBED_DIM, num_layers=DEPTH, 
                              num_heads=NUM_HEADS, mlp_dim=MLP_DIM, 
                              dropout=DROP_RATE)
    
    model = model.to(device)
    #model.load_state_dict(torch.load("working/vit_epoch_80.pt", map_location=device))
    count_parameters(model)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4/2, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    indices = random.sample(range(len(test_dataset)), 9)


    # Train the model
    start = datetime.now()
    print(f"Training started at: {start}")
    train_losses, train_accurs, test_accurs = training_model(model, 
                                                             train_loader, 
                                                             test_loader, 
                                                             optimizer, 
                                                             criterion, 
                                                             scheduler, 
                                                             EPOCHS, 
                                                             start_epoch = 60)
    end = datetime.now()
    print(f"Training finished. Elapsed time: {end - start}")
    
    # =========================
    # Plot Loss vs Train/Test Accuracy
    # =========================
    epochs = range(1, len(train_losses) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Loss (left Y axis)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:red")
    ax1.plot(epochs, train_losses, color="tab:red", label="Train Loss")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    # Accuracy (right Y axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", color="tab:blue")
    ax2.plot(epochs, train_accurs, color="tab:blue", linestyle="--", label="Train Accuracy")
    ax2.plot(epochs, test_accurs, color="tab:green", linestyle="-.", label="Test Accuracy")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    # Legend (merge both axes)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center")

    plt.title("Training Loss vs Train/Test Accuracy")

    # Save or show
    if matplotlib.get_backend() != "agg":
        plt.show()
    else:
        plot_path = os.path.join(PLOT_DIR, "loss_vs_accuracy.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Loss vs Accuracy plot saved to {plot_path}")

    save_path = os.path.join(SAVE_DIR,"model_weights.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")
    
    # Setting up the visualization.
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    fig.tight_layout(pad=3.0)

    for ax, idx in zip(axes.flat, indices):
        x, y = test_dataset[idx]
        x_input = x.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(x_input)
            pred = output.argmax(1).item()

        # Denormalization
        img = denormalize(x).clamp(0, 1).permute(1, 2, 0).cpu().numpy()

        # Display treinamento
        ax.imshow(img)
        color = 'green' if pred == y else 'red'
        ax.set_title(f"Pred: {test_dataset.classes[pred]}", color=color)
        ax.axis('off')

    if matplotlib.get_backend() != "agg":
        plt.show()
    else:
        print("Using non interactive backend, training results saved to file")
        plot_path = os.path.join(PLOT_DIR,"treinamento.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        
    # Ok, model is trained, now inference testing...
    model.eval()
    
    # Estatísticas do treinamento
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = outputs.argmax(1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            
    # Build the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_dataset.classes)
    
    # Custom cmap
    reds_cmap = LinearSegmentedColormap.from_list("custom_reds", ["#FAF9F6", "#B40203"])
    
    # Visualize
    fig, ax = plt.subplots(figsize=(8,8))
    fig.patch.set_facecolor("#FAF9F6")
    disp.plot(ax=ax, cmap=reds_cmap, xticks_rotation=45)
    ax.grid(False)
    plt.title("Confusion Matrix", fontsize=14)
    
    # Show...
    if matplotlib.get_backend() != "agg":
        plt.show()
    else:
        print("Using non interactive backend, confusion matrix saved to file")
        plot_path = os.path.join(PLOT_DIR,"confusion_matrix.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

###

def configs(config_file="config.json"):
    #
    global BATCH_SIZE
    global EPOCHS
    global LEARNING_RATE
    global PATCH_SIZE
    global NUM_CLASSES
    global IMAGE_SIZE
    global CHANNELS
    global EMBED_DIM
    global NUM_HEADS
    global DEPTH
    global MLP_DIM
    global DROP_RATE
    global WORK_DIR
    global SAVE_DIR
    global PLOT_DIR
    #
    try:
        with open(config_file) as cfg:
            cfgs = json.load(cfg)
    except Exception as e:
        print(f"[INFO] failed to open config file {config_file}")
        print(f"[INFO] reason: {str(e)}")
        print("[INFO] Using hardwired default values")
        return True
    for a_key in set(cfgs.keys()):
        if  a_key == "batch_size":
            BATCH_SIZE = cfgs["batch_size"]
            if not  isinstance(BATCH_SIZE, int):
                print(f"[ERROR] invalid value for batch_size: {BATCH_SIZE}")
                return False
            if BATCH_SIZE < 16 or BATCH_SIZE % 16 != 0:
                print(f"[ERROR] batch_size {BATCH_SIZE} is not a multiple of 16")
                return False
        elif a_key == "epochs":
            EPOCHS = cfgs["epochs"]
            if not isinstance(EPOCHS, int) or EPOCHS < 60:
                print(f"[ERROR] invalid number of epochs {EPOCHS}")
                return False
        elif a_key == "learning_rate":
            LEARNING_RATE = cfgs["learning_rate"]
            if not isinstance(LEARNING_RATE, float) or LEARNING_RATE < 0.0 or LEARNING_RATE >= 1.0:
                print(f"[ERROR] invalid learning rate {LEARNING_RATE}")
                return False
        elif a_key == "image_size":
            IMAGE_SIZE = cfgs["image_size"]
            if not isinstance(IMAGE_SIZE, int) or IMAGE_SIZE < 32 or IMAGE_SIZE % 4 != 0:
                print(f"[ERROR] invalid image size {IMAGE_SIZE}")
                return False
        elif a_key == "patch_size":
            PATCH_SIZE = cfgs["patch_size"]
            if not isinstance(PATCH_SIZE, int) or PATCH_SIZE < 4:
                print(f"[ERROR] invalid patch size {PATCH_SIZE}")
                return False
        elif a_key == "num_classes":
            NUM_CLASSES = cfgs["num_classes"]
            if not isinstance(NUM_CLASSES, int) or NUM_CLASSES < 1:
                print(f"[ERROR] invaliid number of classes {NUM_CLASSES}")
                return False
        elif a_key == "channels":
            CHANNELS = cfgs["channels"]
            if not isinstance(CHANNELS, int) or CHANNELS < 1 or CHANNELS > 3:
                print(f"[ERROR] invalid number of channels {CHANNELS}")
                return False
        elif a_key == "embed_dim":
            EMBED_DIM = cfgs["embed_dim"]
            if not isinstance(EMBED_DIM, int) or EMBED_DIM < 1:
                print(f"[ERROR] invalid embed dim {EMBED_DIM}")
                return False
        elif a_key == "num_heads":
            NUM_HEADS = cfgs["num_heads"]
            if not isinstance(NUM_HEADS, int) or NUM_HEADS < 1:
                print(f"[ERROR] invalid num heads {NUM_HEADS}")
                return False
        elif a_key == "depth":
            DEPTH = cfgs["depth"]
            if not isinstance(DEPTH, int) or DEPTH < 1:
                print(f"[ERROR] invalid depth {DEPTH}")
                return False
        elif a_key == "mlp_dim":
            MLP_DIM = cfgs["mlp_dim"]
            if not isinstance(MLP_DIM, int) or MLP_DIM < 1:
                print(f"[ERROR] invalid mlp dim {MLP_DIM}")
                return False
        elif a_key == "drop_rate":
            DROP_RATE = cfgs["drop_rate"]
            if not isinstance(DROP_RATE, float) or DROP_RATE < 0.0 or DROP_RATE >= 1.0:
                print(f"[ERROR] invalid drop rate {DROP_RATE}")
                return False
        elif a_key == "work_dir":
            WORK_DIR = cfgs["work_dir"]
            try:
                if not os.path.exists(WORK_DIR):
                    print(f"[INFO] work dir {WORK_DIR} does not exist, trying to create it")
                    os.mkdir(WORK_DIR)
                else:
                    if not os.path.isdir(WORK_DIR):
                        print(f"[ERROR] {WORK_DIR} exists but is not a directory")
                        return False
            except Exception as e:
                print(f"[ERROR] fail to process work dir {WORK_DIR}")
                print(f"[INFO] reason: {str(e)}")
                return False
        elif a_key == "save_dir":
            SAVE_DIR = cfgs["save_dir"]
            try:
                if not os.path.exists(SAVE_DIR):
                    print(f"[INFO] save dir {SAVE_DIR} does not exist, trying to create it")
                    os.mkdir(SAVE_DIR)
                else:
                    if not os.path.isdir(SAVE_DIR):
                        print(f"[ERROR] {SAVE_DIR} exists but is not a directory")
                        return False
            except Exception as e:
                print(f"[ERROR] fail to process save dir {SAVE_DIR}")
                print(f"[INFO] reason: {str(e)}")
                return False
        elif a_key == "plot_dir":
            PLOT_DIR = cfgs["plot_dir"]
            try:
                if not os.path.exists(PLOT_DIR):
                    print(f"[INFO] plot dir {PLOT_DIR} does not exist, trying to create it")
                    os.mkdir(PLOT_DIR)
                else:
                    if not os.path.isdir(PLOT_DIR):
                        print(f"[ERROR] {PLOT_DIR} exists but is not a directory")
                        return False
            except Exception as e:
                print(f"[ERROR] fail to process plot dir {PLOT_DIR}")
                print(f"[INFO] reason: {str(e)}")
                return False
        else:
            print(f"[ERROR] unknown key {a_key}. Ignored")
    if IMAGE_SIZE % PATCH_SIZE != 0:
        print(f"[ERROR] IMAGE_SIZE % PATCH SIZE should be 0 but... {IMAGE_SIZE % PATCH_SIZE}")
        return False
    return True
        
if __name__ == "__main__":
    if configs():
        main()
