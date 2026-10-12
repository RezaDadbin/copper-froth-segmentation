"""Historical source export; maintained entry points live in scripts/.

Original: codes/UNET/.ipynb_checkpoints/U-Net-Dataloader-checkpoint.ipynb
Importing this file does not train, install packages, rename files or render.
historical_experiment() is a manual research transcript, not a supported pipeline.
"""


def historical_experiment():
    # %% Original cell 0 (code)
    import torch
    import json
    import os 
    import cv2
    import numpy as np
    from PIL import Image
    import tifffile as tiff
    import matplotlib.pyplot as plt

    # %% Original cell 1 (code)
    os.getcwd()

    # %% Original cell 2 (code)
    import os
    import json
    import numpy as np
    import tifffile as tiff
    import cv2
    import torch
    from torch.utils.data import Dataset
    import torchvision.transforms as transforms

    class FrothSegmentationDataLoader(Dataset):
        def __init__(self, data_dir):
            self.data_dir = data_dir
            self.samples = self._load_filenames()
            self.transform = transforms.Compose([
                transforms.CenterCrop(1080),  # Crop to a centered 1080x1080 square
                transforms.Resize((256, 256)),  # Resize to 256x256
                transforms.ToTensor(),  # Convert to tensor
                # transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize
            ])
            self.transform2 = transforms.Compose([
                transforms.CenterCrop(480),  # Crop to a centered 1080x1080 square
                transforms.Resize((256, 256)),  # Resize to 256x256
                transforms.ToTensor(),  # Convert to tensor
                # transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize
            ])

        def _load_filenames(self):
            samples = []
            for file in os.listdir(self.data_dir):
                json_file = None  # Initialize json_file to avoid UnboundLocalError
            
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

            # Load image
            try:
                image = tiff.imread(image_path)
                image = np.array(image, dtype=np.uint8)  # Ensure float32
                height, width = image.shape[:2]
            except Exception as e:
                print(f"Error loading image: {image_path}, Error: {e}")
                return None

            # Ensure grayscale images have 3 channels
            if len(image.shape) == 2:  
                image = np.stack([image] * 3, axis=-1)

            # Create an empty mask
            mask = np.zeros((height, width), dtype=np.uint8)

            # Load JSON and extract polygons
            try:
                with open(json_path, 'r') as f:
                    annotations = json.load(f)
            except json.JSONDecodeError:
                print(f"Error loading JSON file: {json_path}")
                return None

            for shape in annotations['shapes']:
                if shape['label'] == 'froth':
                    points = np.array(shape['points'], dtype=np.int32)
                    cv2.fillPoly(mask, [points], 255)

            # Convert to PIL Image for transformations
            image = transforms.functional.to_pil_image(image)
            mask = transforms.functional.to_pil_image(mask)

            # Apply transformations
            if image.size[1] > 480:
                image = self.transform(image)  # Normalized tensor (3, 256, 256)
                mask = self.transform(mask)  # Normalized tensor (1, 256, 256)
            else:
                image = self.transform2(image)  # Normalized tensor (3, 256, 256)
                mask = self.transform2(mask)  # Normalized tensor (1, 256, 256)

            return image, mask

        def visualize(self, index):
            sample = self.__getitem__(index)
            if sample is not None:
                image, mask = sample
            
                # Convert image and mask to numpy for visualization
                image_np = image.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
                mask_np = mask.squeeze().numpy()  # (1, H, W) -> (H, W)

                # Plot the image and mask
                import matplotlib.pyplot as plt
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

    # Example Usage
    data_loader = FrothSegmentationDataLoader("train1")
    print("Total samples:", len(data_loader))
    data_loader.visualize(0)

    print(data_loader.__getitem__(0)[0].shape)
    print(data_loader.__getitem__(0)[1].shape)

    # %% Original cell 3 (code)
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class UNet(nn.Module):
        def __init__(self):
            super(UNet, self).__init__()

            self.pool = nn.MaxPool2d(2, 2)

            self.enc1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.enc2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.enc3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.enc4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.enc5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)

            self.bottleneck = nn.Conv2d(512, 512, kernel_size=3, padding=1)

            self.upconv5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
            self.dec5 = nn.Conv2d(1024, 256, kernel_size=3, padding=1)

            self.upconv4 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
            self.dec4 = nn.Conv2d(512, 128, kernel_size=3, padding=1)

            self.upconv3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
            self.dec3 = nn.Conv2d(256, 64, kernel_size=3, padding=1)

            self.upconv2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
            self.dec2 = nn.Conv2d(128, 32, kernel_size=3, padding=1)

            self.upconv1 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
            self.dec1 = nn.Conv2d(64, 16, kernel_size=3, padding=1)

            self.final = nn.Conv2d(16, 1, kernel_size=1)

        def forward(self, x):
            # print("x   ",x.shape)
            e1 = F.relu(self.enc1(x))
            # print("e1  ",e1.shape)
            e2 = F.relu(self.enc2(self.pool(e1)))
            # print("e2  ",e2.shape)
            e3 = F.relu(self.enc3(self.pool(e2)))
            # print("e3  ",e3.shape)
            e4 = F.relu(self.enc4(self.pool(e3)))
            # print("e4  ",e4.shape)
            e5 = F.relu(self.enc5(self.pool(e4)))
            # print("e5  ",e5.shape)

            b = F.relu(self.bottleneck(self.pool(e5)))
            # print("b   ",b.shape)
        
            # print("b_up", self.upconv5(b).shape)
            # print("b_t ",(torch.cat([self.upconv5(b), e5], dim=1)).shape)
            d5 = F.relu(self.dec5(torch.cat([self.upconv5(b), e5], dim=1)))
            # print("d5  ",d5.shape)
        
            # print("d5up", self.upconv4(d5).shape)
            # print("d5to",(torch.cat([self.upconv4(d5), e4], dim=1)).shape)
            d4 = F.relu(self.dec4(torch.cat([self.upconv4(d5), e4], dim=1)))
            # print("d4  ",d4.shape)
        
            # print("d4up", self.upconv3(d4).shape)
            # print("d4to",(torch.cat([self.upconv3(d4), e3], dim=1)).shape)
            d3 = F.relu(self.dec3(torch.cat([self.upconv3(d4), e3], dim=1)))
            # print("d3  ",d3.shape)
        
            # print("d3up", self.upconv2(d3).shape)
            # print("d3to",(torch.cat([self.upconv2(d3), e2], dim=1)).shape)
            d2 = F.relu(self.dec2(torch.cat([self.upconv2(d3), e2], dim=1)))
            # print("d2  ",d2.shape)
        
            # print("d2up", self.upconv1(d2).shape)
            # print("d2to",(torch.cat([self.upconv1(d2), e1], dim=1)).shape)
            d1 = F.relu(self.dec1(torch.cat([self.upconv1(d2), e1], dim=1)))
            # print("d1  ",d1.shape)
        
            # print("finl", self.final(d1).shape)
            return self.final(d1)

    # Example usage
    model = UNet()
    model(torch.randn(1, 3, 256, 256)).shape # Output: torch.Size([1, 1, 256, 256])

    # %% Original cell 4 (code)
    model

    # %% Original cell 5 (code)
    # Configurations
    train_data_dir = "train1"  # Path to your training dataset
    val_data_dir = "test1"  # Path to your validation dataset
    fin_data_dir = "eval"
    batch_size = 2
    learning_rate = 0.00001
    num_epochs = 300
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # Create the dataset and DataLoaders
    train_dataset = FrothSegmentationDataLoader(train_data_dir)
    val_dataset = FrothSegmentationDataLoader(val_data_dir)
    fin_dataset = FrothSegmentationDataLoader(fin_data_dir)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    fin_loader = torch.utils.data.DataLoader(fin_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer & Loss
    model = UNet().to(device)  # Make sure to move the model to the correct device
    loss_fn = nn.BCEWithLogitsLoss()  # Binary segmentation task
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Optional: Print dataset info
    print("Total samples in training dataset:", len(train_dataset))
    print("Total samples in validation dataset:", len(val_dataset))
    print("Total smaples in evaluation dataset:", len(fin_dataset))

    # %% Original cell 6 (code)
    def plot_metrics(losses_train, losses_val, accuracies_train, accuracies_val):
        # clear_output(wait=True)
        plt.figure(figsize=(14, 5))  # Wider layout for better visualization

        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(losses_train, label='Loss Train', color='blue')
        plt.plot(losses_val, label='Loss Validation', color='orange')
    
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training & Validation Loss')
        plt.legend()

        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(accuracies_train, label='Pixel Accuracy Train', color='blue')
        plt.plot(accuracies_val, label='Pixel Accuracy Validation', color='green')
    
        plt.xlabel('Epoch')
        plt.ylabel('Pixel Accuracy')
        plt.title('Training & Validation Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()

    # %% Original cell 7 (code)
    # Training Loop
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct_pixels_train = 0
        total_pixels_train = 0
    
        running_val_loss = 0.0
        correct_pixels_val = 0
        total_pixels_val = 0
    

  

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = loss_fn(outputs.squeeze(), masks.squeeze())  
            running_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()

            # Compute pixel accuracy
            # with torch.no_grad()
            predictions = (outputs > 0.7).float()  # Binarize predictions
            correct_pixels_train += (predictions == masks).sum().item()
            # print("correct",xa correct_pixels_train)
            total_pixels_train += masks.numel()

        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = correct_pixels_train / total_pixels_train
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        # Validation Phase
        model.eval()  
    

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)

                outputs = model(images)

                # Compute loss
                loss = loss_fn(outputs.squeeze(), masks.squeeze())
                running_val_loss += loss.item()

                # Compute pixel accuracy
                predictions = (outputs > 0.7).float()
            
                correct_pixels_val += (predictions == masks).sum().item()
                total_pixels_val += masks.numel()


        avg_val_loss = running_val_loss / len(val_loader)
        val_accuracy = correct_pixels_val / total_pixels_val
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        # Print progress
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")

        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")

        # Plot metrics
        plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)

    # %% Original cell 8 (code)
    torch.save(model.state_dict(), 'unet_froth_segmentation_300epochs_iou81.pth')  

    # %% Original cell 9 (code)
    model.load_state_dict(torch.load('unet_froth_segmentation_300epochs_iou81.pth'))

    # %% Original cell 10 (code)
    model.eval()  

    output_dir = "output_images_viridis"
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for images, masks in fin_loader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)

            for i in range(outputs.size(0)):  # Iterate over each sample in the batch
                output_image = outputs[i].detach().cpu().numpy()  # Get ith image from the batch
        
                # Normalize output to be between 0 and 1
                output_image = output_image - output_image.min()
                output_image = output_image / output_image.max()
            
                # Plot the output
                plt.imshow(output_image[0], cmap='twilight_shifted')  # Use grayscale colormap
                plt.show()


            # output_filename = os.path.join(output_dir, f"output_{i}_{i}.png")
            # plt.imsave(output_filename, output_image[0], cmap='biridis')
            # # Compute loss
        
            # loss = loss_fn(outputs.squeeze(), masks.squeeze())
            # running_val_loss += loss.item()

            # # Compute pixel accuracy
            # predictions = (outputs > 0.7).float()
            
            # correct_pixels_val += (predictions == masks).sum().item()
            # total_pixels_val += masks.numel()
        

    # %% Original cell 11 (code)
    import os
    import torch
    import matplotlib.pyplot as plt
    import numpy as np # Make sure numpy is imported

    # Ensure your model, fin_loader, and device are defined
    # model = YourModel()
    # fin_loader = YourDataLoader()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    output_dir = "output_images_viridis"
    os.makedirs(output_dir, exist_ok=True)

    sample_counter = 0  # Initialize a counter for unique filenames

    with torch.no_grad():
        for images, masks in fin_loader: # Loop over batches
            images, masks = images.to(device), masks.to(device)
            outputs = model(images) # Model outputs for the batch

            for i in range(outputs.size(0)):  # Loop over each sample in the batch
                # Get the i-th image tensor from the batch outputs
                # outputs[i] typically has shape (channels, height, width)
                single_output_tensor = outputs[i]

                # Detach from graph, move to CPU, and convert to NumPy array
                output_image_numpy = single_output_tensor.detach().cpu().numpy()

                # Normalize output to be between 0 and 1
                min_val = output_image_numpy.min()
                max_val = output_image_numpy.max()

                if max_val > min_val:  # Avoid division by zero if the image is constant
                    normalized_image = (output_image_numpy - min_val) / (max_val - min_val)
                else:
                    # If min_val == max_val, the image is constant (e.g., all black).
                    # Normalize to a zero image.
                    normalized_image = np.zeros_like(output_image_numpy)

                # Your original code uses output_image[0], suggesting you want to save
                # the first channel if the output is multi-channel (e.g., C, H, W).
                # If your output is already single-channel (H, W), you can remove the [0].
                image_to_save = normalized_image[0] if normalized_image.ndim == 3 else normalized_image

                # Optional: Display the output (can be slow/overwhelming for many images)
                # plt.imshow(image_to_save, cmap='copper')
                # plt.title(f"Output Sample {sample_counter}")
                # plt.show() # This will block execution until the plot window is closed

                # Save the processed output image
                # Use the sample_counter for a unique filename for each image
                output_filename = os.path.join(output_dir, f"output_{sample_counter}.png")
                plt.imsave(output_filename, image_to_save, cmap='viridis') # Using 'grey' as in your original save line

                sample_counter += 1 # Increment for the next sample

    print(f"Saved {sample_counter} images to {output_dir}")

    # %% Original cell 12 (code)

    pass


if __name__ == "__main__":
    raise SystemExit("Historical transcript only. Use the linked scripts/ commands in README.md.")
