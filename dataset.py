import re
import os
import numpy as np
from torch.utils.data import Dataset
import cv2
import torch

class Doodle(Dataset): 
    def __init__(self, root_path, image_to_take_ratio = 0.3, split_ratio = 0.8, mode = "train"):
        self.root_path = root_path
        self.classes = []
        self.images = []
        self.labels = []

        for iter, f in enumerate(os.listdir(self.root_path)):
            #Check file validation
            if not f.endswith(".npy"):
                continue

            #Extract the class name from the file name
            class_name = self.__extract_class_name(f)
            if class_name is None:
                continue
            self.classes.append(class_name)
            
            #Read the npy file
            npy_path = os.path.join(self.root_path, f)
            data = np.load(npy_path)
            total_images = len(data)

            #Take some images from the npy file uniformly by image_to_take_ratio
            num_image_to_take = int(total_images * image_to_take_ratio)    
            if num_image_to_take > 0:
                stride = total_images / num_image_to_take
                # Get indices of images to select (uniformly distributed)
                indices = np.floor(np.arange(0, total_images, stride)).astype(int)
                # Ensure we don't exceed the array bounds
                indices = indices[:num_image_to_take]
                # Extract the selected images - use NumPy indexing for better performance
                selected_data = data[indices]
            else:
                selected_data = np.array([])

            # Split data into train and test
            num_selected = len(selected_data)
            split_index = int(num_selected * split_ratio)
            
            # Make sure split_index is valid (at least 1 for train mode)
            if mode == "train":
                self.images.extend(selected_data[:split_index])
                self.labels.extend([iter] * split_index)
            elif mode == "test":
                self.images.extend(selected_data[split_index:])
                self.labels.extend([iter] * (num_selected - split_index))
      
    """Extract the category name from the numpy folder"""
    def __extract_class_name(self, file_name):
        match = re.search(r"bitmap_(\w+)", file_name)
        return match.group(1) if match else None

    def __len__(self):
        """Return the total number of samples"""
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32)
        image = image.reshape(28, 28)
        image = cv2.resize(image, (224, 224))
        image /= 255.0
        image = torch.from_numpy(image).float().unsqueeze(0) 

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return image, label
        

if __name__ == "__main__":
    dataset = Doodle(root_path = "./datasets/doodles", mode = "train")
    print(f"Dataset size: {len(dataset)}")
    print(f"Classes: {dataset.classes}")
    
    if len(dataset) > 0:
        # Make sure we don't try to access an index that doesn't exist
        sample_idx = min(10001, len(dataset) - 1)
        img, label = dataset[sample_idx]
        print(f"Image shape: {img.shape}")
        print(f"Class: {dataset.classes[label]}")