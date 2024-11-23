import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt

class PseudoAnnotationDataset(Dataset):
    def __init__(self, image_paths, mask_paths=None, pseudo_labels=None, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths or [None] * len(image_paths)
        self.pseudo_labels = pseudo_labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        # Handle ground truth or pseudo labels
        if self.mask_paths[idx] is not None:
            mask = Image.open(self.mask_paths[idx]).convert('L')
            mask = transforms.ToTensor()(mask)
            mask = (mask > 0.5).float()  # Binarize the mask
        elif self.pseudo_labels is not None:
            mask = self.pseudo_labels[idx]
        else:
            mask = torch.zeros_like(image[0])  # Default empty mask

        return image, mask

class UNetWithPseudoAnnotation(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        # Encoder layers
        self.enc1 = self._double_conv(in_channels, 64)
        self.enc2 = self._double_conv(64, 128)
        self.enc3 = self._double_conv(128, 256)
        self.enc4 = self._double_conv(256, 512)
        
        # Bottleneck layer
        self.bottleneck = self._double_conv(512, 1024)

        # Decoder layers
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self._double_conv(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._double_conv(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._double_conv(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._double_conv(128, 64)

        # Final convolution layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def _double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder path
        enc1 = self.enc1(x)
        enc2 = self.enc2(nn.MaxPool2d(2)(enc1))
        enc3 = self.enc3(nn.MaxPool2d(2)(enc2))
        enc4 = self.enc4(nn.MaxPool2d(2)(enc3))

        # Bottleneck path
        bottleneck = self.bottleneck(nn.MaxPool2d(2)(enc4))

        # Decoder path
        dec4 = torch.cat((enc4, self.upconv4(bottleneck)), dim=1)
        dec4 = self.dec4(dec4)

        dec3 = torch.cat((enc3, self.upconv3(dec4)), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = torch.cat((enc2, self.upconv2(dec3)), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = torch.cat((enc1, self.upconv1(dec2)), dim=1)
        dec1 = self.dec1(dec1)

        return torch.sigmoid(self.final_conv(dec1))

def generate_pseudo_labels(model, unlabeled_loader, device):
    """ Generate pseudo labels for unlabeled images """
    model.eval()
    pseudo_labels = []
    confident_indices = []

    with torch.no_grad():
        for idx, (images, _) in enumerate(unlabeled_loader):
            images = images.to(device)
            outputs = model(images)

            # Convert to binary mask and assess confidence
            binary_masks = (outputs > 0.5).float()
            
            # Calculate confidence scores properly
            confidence_scores = torch.max(outputs, dim=2)[0]  # max over height
            confidence_scores = torch.max(confidence_scores, dim=2)[0]  # max over width
            confidence_scores = torch.mean(confidence_scores, dim=1)  # mean over channels

            for j in range(len(binary_masks)):
                if confidence_scores[j].item() >= 0.7:
                    pseudo_labels.append(binary_masks[j].cpu())
                    confident_indices.append(idx * unlabeled_loader.batch_size + j)

    return pseudo_labels, confident_indices

def create_directories(base_dir):
    """ Create necessary directories for the project """
    os.makedirs(os.path.join(base_dir,'data/labeled_images'), exist_ok=True)
    os.makedirs(os.path.join(base_dir,'data/labeled_masks'), exist_ok=True)
    os.makedirs(os.path.join(base_dir,'data/unlabeled_images'), exist_ok=True)
    os.makedirs(os.path.join(base_dir,'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(base_dir,'results/predictions'), exist_ok=True)
    os.makedirs(os.path.join(base_dir,'results/visualizations'), exist_ok=True)

def save_image(tensor, path):
    """ Save a tensor as an image. """
    image = tensor.squeeze().cpu().numpy()  # Remove channels if single-channel
    image = (image * 255).astype(np.uint8)  # Convert to uint8
    img = Image.fromarray(image)
    img.save(path)

def plot_confusion_matrix(y_true, y_pred):
    """ Plot confusion matrix """
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()

def main():
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    base_dir = '/home/mhpromit7473/Documents/melanoma-pseudo-annotation'
    
    # Create necessary directories
    create_directories(base_dir)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths configuration
    labeled_image_dir = os.path.join(base_dir,'data/labeled_images')
    labeled_mask_dir = os.path.join(base_dir,'data/labeled_masks')
    unlabeled_image_dir = os.path.join(base_dir,'data/unlabeled_images')

    # Load labeled and unlabeled image paths
    labeled_image_paths = sorted([os.path.join(labeled_image_dir,f) for f in os.listdir(labeled_image_dir) if f.endswith('.png')])
    labeled_mask_paths = sorted([os.path.join(labeled_mask_dir,f) for f in os.listdir(labeled_mask_dir) if f.endswith('.png')])
    unlabeled_image_paths = sorted([os.path.join(unlabeled_image_dir,f) for f in os.listdir(unlabeled_image_dir) if f.endswith('.png')])

    # Transformations setup
    transform = transforms.Compose([
       transforms.Resize((256, 256)),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Split labeled data into training and validation sets 
    train_image_paths, val_image_paths, train_mask_paths, val_mask_paths = train_test_split(
        labeled_image_paths, labeled_mask_paths, test_size=0.2, random_state=42
    )

    # Initial training with labeled data 
    train_dataset = PseudoAnnotationDataset(train_image_paths, train_mask_paths, transform=transform) 
    val_dataset = PseudoAnnotationDataset(val_image_paths, val_mask_paths, transform=transform) 

    # Reduced batch size to prevent OOM
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True) 
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False) 

    # Model and training setup 
    model = UNetWithPseudoAnnotation().to(device) 
    criterion = nn.BCELoss() 
    optimizer = optim.Adam(model.parameters(), lr=1e-4) 

    best_val_loss = float('inf') 
    num_epochs = 20
    accumulation_steps = 2  # Gradient accumulation steps ```python
    for epoch in range(num_epochs): 
        print(f"Epoch [{epoch + 1}/{num_epochs}]") 
        model.train() 
        epoch_loss = 0 

        optimizer.zero_grad()
        for batch_idx, (images, masks) in enumerate(train_loader): 
            images, masks = images.to(device), masks.to(device) 

            outputs = model(images) 
            loss = criterion(outputs, masks)
            loss = loss / accumulation_steps  # Scale loss for accumulation
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * accumulation_steps  # Scale back for logging

            if (batch_idx + 1) % 5 == 0: 
                print(f" Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}") 

        avg_epoch_loss = epoch_loss / len(train_loader) 
        print(f" Average Epoch Loss: {avg_epoch_loss:.4f}") 

        print("Generating Pseudo Labels...") 
        pseudo_labels, confident_indices = generate_pseudo_labels(model, val_loader, device) 

        print(f" Confident Pseudo Labels: {len(confident_indices)}") 

        if pseudo_labels: 
            pseudo_dataset = PseudoAnnotationDataset(
                [unlabeled_image_paths[i] for i in confident_indices], 
                pseudo_labels=pseudo_labels,
                transform=transform
            )

            # Save the pseudo-label images
            for idx, pseudo_label in enumerate(pseudo_labels):
                save_image(pseudo_label, os.path.join(base_dir, 'results/predictions', f'pseudo_label_{confident_indices[idx]}.png'))

            pseudo_loader = DataLoader(pseudo_dataset, batch_size=16, shuffle=True) 

            for images, masks in pseudo_loader: 
                images, masks = images.to(device), masks.to(device) 

                optimizer.zero_grad() 
                outputs = model(images) 

                pseudo_loss = criterion(outputs, masks) 
                pseudo_loss.backward() 
                optimizer.step() 

        # Save the best model based on validation loss  
        if avg_epoch_loss < best_val_loss: 
            best_val_loss = avg_epoch_loss 
            checkpoint_path = os.path.join(base_dir,'checkpoints/melanoma_pseudo_annotation_model.pth') 
            torch.save(model.state_dict(), checkpoint_path)  
            print(f"Model saved at {checkpoint_path} with validation loss: {best_val_loss:.4f}")

        # Evaluate on validation set 
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():  
            for val_images, val_masks in val_loader:
                val_images, val_masks = val_images.to(device), val_masks.to(device)  
                preds = (model(val_images) > 0.5).cpu().numpy()
                targets = val_masks.cpu().numpy()

                all_preds.append(preds)
                all_targets.append(targets)

                # Save predictions as images
                for i in range(preds.shape[0]):
                    save_image(preds[i], os.path.join(base_dir, 'results/predictions', f'pred_{len(all_preds)*val_loader.batch_size + i}.png'))

            # Flatten predictions and targets
            all_preds = np.concatenate(all_preds).flatten()
            all_targets = np.concatenate(all_targets).flatten()

            # Plot confusion matrix
            plot_confusion_matrix(all_targets, all_preds)

if __name__ == '__main__':
   main()