import re
import os
import numpy as np
from torch.utils.data import Dataset
import cv2
import torch

class Doodle(Dataset): 
    def __init__(self, root_path, split_ratio = 0.8, mode = "train", image_size = 28):
        self.image_size = image_size
        self.classes = []
        self.images = []
        self.labels = []

        for iter, f in enumerate(os.listdir(root_path)):
            #Check file validation
            if not f.endswith(".npy"):
                continue

            #Extract the class name from the file name
            class_name = self.__extract_class_name(f)
            if class_name is None:
                continue
            self.classes.append(class_name)
            
            #Read the npy file
            npy_path = os.path.join(root_path, f)
            data = np.load(npy_path)
            total_images = len(data)
            

            # Split data into train and test
            split_index = int(total_images * split_ratio)
            
            # Make sure split_index is valid (at least 1 for train mode)
            if mode == "train":
                self.images.extend(data[:split_index])
                self.labels.extend([iter] * split_index)
            elif mode == "test":
                self.images.extend(data[split_index:])
                self.labels.extend([iter] * (total_images - split_index))
      
    """Extract the category name from the numpy folder"""
    def __extract_class_name(self, file_name):
        match = re.search(r"bitmap_([\w\- ]+)", file_name)
        return match.group(1) if match else None

    def __len__(self):
        """Return the total number of samples"""
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32)
        image = image.reshape(self.image_size, -1)
        image /= 255.0
        image = torch.from_numpy(image).float().unsqueeze(0) 

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return image, label
        

if __name__ == "__main__":
    dataset = Doodle(root_path = "./datasets/doodles", mode = "train")
    # print(f"Dataset size: {len(dataset)}")
    # print(f"Classes: {dataset.classes}")
    
    if len(dataset) > 0:
        # Make sure we don't try to access an index that doesn't exist
        sample_idx = min(1420000, len(dataset) - 1)
        img, label = dataset[sample_idx]
        cv2.imshow("Sample Image", img)
        print(f"Label: {dataset.classes[label.item()]}")
        cv2.waitKey(0)