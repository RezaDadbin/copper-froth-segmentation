"""Historical source export; maintained entry points live in scripts/.

Original: codes/FCN-RESNET/.ipynb_checkpoints/FCN_RESNET-checkpoint.ipynb
Importing this file does not train, install packages, rename files or render.
historical_experiment() is a manual research transcript, not a supported pipeline.
"""


def historical_experiment():
    # %% Original cell 0 (code)
    import torch
    from torch.utils.data import Dataset
    import torchvision.transforms as transforms
    import torch.nn as nn
    import torch.nn.functional as F
    import json
    import os 
    import cv2
    import numpy as np
    from PIL import Image
    import tifffile as tiff
    import matplotlib.pyplot as plt

    class FrothSegmentationDataLoader(Dataset):
        def __init__(self, data_dir):
            self.data_dir = data_dir
            self.samples = self._load_filenames()
            self.transform = transforms.Compose([
                transforms.CenterCrop(1080),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])
            self.transform2 = transforms.Compose([
                transforms.CenterCrop(480),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])

        def _load_filenames(self):
            samples = []
            for file in os.listdir(self.data_dir):
                json_file = None
                if file.endswith(".tiff"):
                    json_file = file.replace(".tiff", ".json")
                elif file.endswith(".tif"):
                    json_file = file.replace(".tif", ".json")
                if json_file in os.listdir(self.data_dir):
                    samples.append((file, json_file))
            return samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            image_file, json_file = self.samples[index]
            image_path = os.path.join(self.data_dir, image_file)
            json_path = os.path.join(self.data_dir, json_file)

            try:
                image = tiff.imread(image_path)
                image = np.array(image, dtype=np.uint8)
                height, width = image.shape[:2]
            except Exception as e:
                print(f"Error loading image: {image_path}, Error: {e}")
                return None

            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)

            mask = np.zeros((height, width), dtype=np.uint8)
            try:
                with open(json_path, 'r') as f:
                    annotations = json.load(f)
            except Exception as e:
                print(f"Error loading JSON file: {json_path}, Error: {e}")
                return None

            for shape in annotations['shapes']:
                if shape['label'] == 'froth':
                    points = np.array(shape['points'], dtype=np.int32)
                    cv2.fillPoly(mask, [points], 255)

            image = transforms.functional.to_pil_image(image)
            mask = transforms.functional.to_pil_image(mask)

            if image.size[1] > 480:
                image = self.transform(image)
                mask = self.transform(mask)
            else:
                image = self.transform2(image)
                mask = self.transform2(mask)

            return image, mask

        def visualize(self, index):
            sample = self.__getitem__(index)
            if sample is not None:
                image, mask = sample
                image_np = image.permute(1, 2, 0).numpy()
                mask_np = mask.squeeze().numpy()

                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.imshow(image_np)
                plt.title('Image')
                plt.axis('off')
                plt.subplot(1, 2, 2)
                plt.imshow(mask_np, cmap='gray')
                plt.title('Mask')
                plt.axis('off')
                plt.show()
            else:
                print("Failed to load sample for visualization.")

    # %% Original cell 1 (code)
    import torchvision

    # Load pretrained FCN with ResNet50 backbone
    def get_fcn_model(pretrained=True):
        model = torchvision.models.segmentation.fcn_resnet50(pretrained=pretrained)
        # Replace classifier head for binary segmentation
        model.classifier[4] = nn.Conv2d(512, 1, kernel_size=1)
        return model

    # %% Original cell 2 (code)
    train_data_dir = "../train1"
    val_data_dir = "../test1"
    batch_size = 4
    learning_rate = 1e-4
    num_epochs = 50
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    train_dataset = FrothSegmentationDataLoader(train_data_dir)
    val_dataset = FrothSegmentationDataLoader(val_data_dir)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = get_fcn_model(pretrained=True).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Total training samples:", len(train_dataset))
    print("Total validation samples:", len(val_dataset))

    # %% Original cell 3 (code)
    def compute_pixel_accuracy(pred_mask, gt_mask):
        total = gt_mask.numel()
        correct = (pred_mask == gt_mask).sum().item()
        return correct / total

    def plot_metrics(losses_train, losses_val, accuracies_train, accuracies_val):
        plt.figure(figsize=(14, 5))
        plt.subplot(1, 2, 1)
        plt.plot(losses_train, label='Train Loss')
        plt.plot(losses_val, label='Val Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss'); plt.legend()
    
        plt.subplot(1, 2, 2)
        plt.plot(accuracies_train, label='Train Acc')
        plt.plot(accuracies_val, label='Val Acc')
        plt.xlabel('Epoch'); plt.ylabel('Pixel Accuracy'); plt.title('Accuracy'); plt.legend()
        plt.tight_layout()
        plt.show()

    # %% Original cell 4 (code)
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
    
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)['out']  # FCN returns dict with 'out'
            loss = loss_fn(outputs.squeeze(), masks.squeeze())
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct_train += (predictions == masks).sum().item()
            total_train += masks.numel()
    
        avg_train_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)
    
        # validation
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)['out']
                loss = loss_fn(outputs.squeeze(), masks.squeeze())
                running_val_loss += loss.item()
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                correct_val += (predictions == masks).sum().item()
                total_val += masks.numel()
    
        avg_val_loss = running_val_loss / len(val_loader)
        val_acc = correct_val / total_val
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)
    
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        plot_metrics(train_losses, val_losses, train_accs, val_accs)

    # %% Original cell 5 (code)
    import torch
    from pathlib import Path
    import os 
    print(os.getcwd())

    save_path = Path("checkpoints"); save_path.mkdir(parents=True, exist_ok=True)

    # after training finishes
    model_cpu = model.to("cpu")        # make device-agnostic
    torch.save(model_cpu.state_dict(), save_path / "model_weights.pth")

    # %% Original cell 6 (code)

    pass


if __name__ == "__main__":
    raise SystemExit("Historical transcript only. Use the linked scripts/ commands in README.md.")
