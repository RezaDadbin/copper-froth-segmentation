"""Historical source export; maintained entry points live in scripts/.

Original: codes/SBSNET/Dataloader (2).ipynb
Importing this file does not train, install packages, rename files or render.
historical_experiment() is a manual research transcript, not a supported pipeline.
"""


def historical_experiment():
    # %% Original cell 0 (code)
    import json
    import torch
    from torch.utils.data import Dataset
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    from PIL import Image
    import os
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from tqdm import tqdm
    exec('from SBSNet import *', globals())

    # %% Original cell 1 (code)
    import os
    import json
    import numpy as np
    from PIL import Image
    import cv2
    import torch
    from torch.utils.data import Dataset

    class PolygonDataset(Dataset):
        def __init__(self, image_folder, transform=None, image_size=(256, 256)):
            self.image_folder = image_folder
            self.transform = transform
            self.image_size = image_size
            self.image_files = [f for f in os.listdir(image_folder) if f.endswith('.tiff')]
            self.label_files = [f.replace('.tiff', '.json') for f in self.image_files]

        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, idx):
            image_path = os.path.join(self.image_folder, self.image_files[idx])
            json_path = os.path.join(self.image_folder, self.label_files[idx])
        
            # Open image and get its original size
            image = Image.open(image_path).convert('RGB')
            original_size = image.size  # (width, height)
    
            # Compute crop box for center crop (square 1080x1080)
            crop_size = 1080
            left = (original_size[0] - crop_size) // 2
            upper = (original_size[1] - crop_size) // 2
            right = left + crop_size
            lower = upper + crop_size
    
            # Load the mask data from JSON
            with open(json_path, 'r') as f:
                data = json.load(f)
    
            # Create a mask with the cropped size (1080x1080)
            mask = np.zeros((1080, 1080), dtype=np.float32)
    
            # Adjust polygon points for the cropped region
            for shape in data['shapes']:
                points = np.array(shape['points'], dtype=np.int32)
                points[:, 0] -= left  # Adjust x-coordinates
                points[:, 1] -= upper  # Adjust y-coordinates
                cv2.fillPoly(mask, [points], 1)
    
            # Perform center crop on the image
            image = image.crop((left, upper, right, lower))
    
            # Convert mask back to PIL Image for resizing
            mask = Image.fromarray(mask)
            mask = mask.resize(self.image_size, resample=Image.BILINEAR)
    
            # Convert the resized mask to a tensor and add a channel dimension
            mask = torch.tensor(np.array(mask), dtype=torch.float32).unsqueeze(0)
    
            # Resize the cropped image to the target size
            image = image.resize(self.image_size, resample=Image.BILINEAR)
    
            # Apply transformations to the image if provided
            if self.transform:
                image = self.transform(image)
    
            return image, mask

    # %% Original cell 2 (code)
    transform = transforms.Compose([
        # transforms.CenterCrop(1080),
        # transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # %% Original cell 3 (code)
    # Access the dataset
    dataset = PolygonDataset(image_folder='./', transform=transform)

    # Get an item
    image, mask = dataset[0]

    # Check the types
    print(type(image))  # Should already be <class 'torch.Tensor'>
    print(type(mask))   # Should also be <class 'torch.Tensor'>

    # If you want to verify the tensor shape
    print(image.shape)  # Should be (C, H, W), e.g., (3, 256, 256)
    print(mask.shape)   # Should be (1, H, W), e.g., (1, 256, 256)

    # If you want to visualize the image, convert it back to a NumPy array
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as F

    plt.imshow(F.to_pil_image(image))  # Visualizes the image tensor
    plt.show()

    plt.imshow(mask.squeeze(0), cmap='gray')  # Visualizes the mask tensor
    plt.show()

    # %% Original cell 4 (code)

    # %% Original cell 5 (code)

    # %% Original cell 6 (raw)
    # !python SBSNet.py

    # %% Original cell 7 (code)

    # %% Original cell 8 (code)

    # %% Original cell 9 (code)

    # %% Original cell 10 (code)

    # %% Original cell 11 (code)
    dataset = PolygonDataset(image_folder='./train', transform=transform)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    test_dataset = PolygonDataset(image_folder='./test', transform=transform)
    data_loader_test = DataLoader(dataset, batch_size=2, shuffle=True)

    # %% Original cell 12 (code)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SBSNet(num_classes=2)  # replace with your actual model
    model = model.to(device)  # Move the model to GPU if available

    optimizer = optim.Adam(model.parameters(), lr=0.0001, )
    criterion = nn.BCEWithLogitsLoss()

    num_epochs = 50

    # %% Original cell 13 (code)
    losses_train = []
    accuracies_train = []

    losses_test = []
    accuracies_test = []

    # %% Original cell 14 (code)
    for epoch in range(num_epochs):
        model.train()
        running_loss_train = 0.0
        running_loss_test = 0.0
    
        total_pixels_train = 0
        total_pixels_test = 0
    
        correct_pixels_train = 0
        correct_pixels_test = 0

    
        for images, masks in tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            images = images.to(device)
            masks = masks.to(device)
        
            # Expand mask to match model's output channels (2 channels)
            masks = masks.repeat(1, 2, 1, 1)  # Repeat along the channel dimension
    
            optimizer.zero_grad()
            outputs = model(images)

            # Calculate pixel accuracy
            with torch.no_grad():
                predictions = (torch.sigmoid(outputs) > 0.5).float()  # Binarize predictions
                correct_pixels_train += (predictions == masks).sum().item()
                total_pixels_train += masks.numel()
        
            # Optionally visualize outputs (remove in production for speed)
            # for i in range(outputs.size(0)):
            #     output_image = outputs[i].detach().cpu().numpy()
            #     output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min())
            #     plt.imshow(output_image[0], cmap='gray')
            #     plt.show()

            # Compute loss
            loss_train = criterion(outputs, masks)
            running_loss_train += loss_train.item()
        
            loss_train.backward()
            optimizer.step()




    
        for images, masks in tqdm(data_loader_test, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            images = images.to(device)
            masks = masks.to(device)
        
            # Expand mask to match model's output channels (2 channels)
            masks = masks.repeat(1, 2, 1, 1)  # Repeat along the channel dimension
    
            optimizer.zero_grad()
            outputs = model(images)

            # Calculate pixel accuracy
            with torch.no_grad():
                predictions = (torch.sigmoid(outputs) > 0.5).float()  # Binarize predictions
                correct_pixels_test += (predictions == masks).sum().item()
                total_pixels_test += masks.numel()
        
            # Optionally visualize outputs (remove in production for speed)
            # for i in range(outputs.size(0)):
            #     output_image = outputs[i].detach().cpu().numpy()
            #     output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min())
            #     plt.imshow(output_image[0], cmap='gray')
            #     plt.show()

            # Compute loss
            loss_test = criterion(outputs, masks)
            running_loss_test += loss_test.item()
        
            # loss.backward()
            # optimizer.step()
        
    

        avg_loss_train = running_loss_train / len(data_loader)
        accuracy_train = correct_pixels_train / total_pixels_train
        losses_train.append(avg_loss_train)
        accuracies_train.append(accuracy_train)

        avg_loss_test = running_loss_test / len(data_loader_test)
        accuracy_test = correct_pixels_test / total_pixels_test
        losses_test.append(avg_loss_test)
        accuracies_test.append(accuracy_test)


        plt.figure(figsize=(12, 5))

        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(losses_train, label='Loss Train', color='blue')
        plt.plot(losses_test, label='Loss Test', color='orange')
    
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
    
        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(accuracies_train, label='Pixel Accuracy Train', color='blue')
        plt.plot(accuracies_test, label='Pixel Accuracy Test', color='orange')
    
        plt.xlabel('Epoch')
        plt.ylabel('Pixel Accuracy')
        plt.title('Training Accuracy')
        plt.legend()
    
        plt.tight_layout()
        plt.show()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss_train:.4f}, Pixel Accuracy: {accuracy_train:.4f}")

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")

    print("Training complete.")

    # %% Original cell 15 (code)

    # %% Original cell 16 (code)


    # %% Original cell 17 (code)

    # %% Original cell 18 (code)
    # for images, masks in tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
    #     images = images.to(device)
    #     masks = masks.to(device)
    
    #     # Expand mask to match model's output channels (2 channels)
    #     masks = masks.repeat(1, 2, 1, 1)  # Repeat along the channel dimension

    #     optimizer.zero_grad()
    #     outputs = model(images)
    
    #     for i in range(outputs.size(0)):  # Iterate over each sample in the batch
    #         output_image = outputs[i].detach().cpu().numpy()  # Get ith image from the batch
        
    #         # Normalize output to be between 0 and 1
    #         output_image = output_image - output_image.min()
    #         output_image = output_image / output_image.max()
        
    #         # Plot the output
    #         plt.imshow(output_image[0], cmap='gray')  # Use grayscale colormap
    #         plt.show()

    # %% Original cell 19 (code)

    pass


if __name__ == "__main__":
    raise SystemExit("Historical transcript only. Use the linked scripts/ commands in README.md.")
