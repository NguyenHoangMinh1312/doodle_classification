"""This script is to train the doodle classification model"""
from dataset import Doodle
import torch
import os
from torch.utils.data import DataLoader
from model import DoodleCNN
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import shutil
import matplotlib.pyplot as plt
import numpy as np

class DoodleTrainer:
    def __init__(self, num_epochs = 100, batch_size = 32, learning_rate = 1e-3, img_size = 28,
                 resume_training = False, patience = 10, split_ratio = 0.8,
                 log_path = "./doodle_classification/tensorboard", 
                 checkpoint_path = "./doodle_classification/checkpoint", 
                 data_path = "./datasets/doodles"):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.img_size = img_size
        self.resume_training = resume_training
        self.patience = patience
        self.split_ratio = split_ratio
        self.log_path = log_path
        self.checkpoint_path = checkpoint_path
        self.data_path = data_path

        if self.resume_training:
            checkpoint = os.path.join(self.checkpoint_path, "last.pt")
            try:
                self.saved_data = torch.load(checkpoint, weights_only = False)
            except FileNotFoundError:
                print(f"Checkpoint not found!")
    
    def plot_confusion_matrix(self, writer, cm, class_names, epoch):
        """
        Returns a matplotlib figure containing the plotted confusion matrix.
        """

        figure = plt.figure(figsize=(20, 20))
        # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
        plt.imshow(cm, interpolation='nearest', cmap="hsv")
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        # Normalize the confusion matrix.
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 2.

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm[i, j] > threshold else "black"
                plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        writer.add_figure('confusion_matrix', figure, epoch)
        plt.close(figure)  

    def __setUpEnvironment(self):
        #create the log and checkpoint directories
        if self.resume_training == False:
            if os.path.isdir(self.log_path):
                shutil.rmtree(self.log_path)
            os.makedirs(self.log_path)

            if os.path.isdir(self.checkpoint_path):
                shutil.rmtree(self.checkpoint_path)
            os.makedirs(self.checkpoint_path)
        
        #Set up the TensorBoard writer
        self.writer = SummaryWriter(self.log_path)
    
    def __setUpData(self):
        #preprocess the data
        self.train_set = Doodle(root_path = self.data_path,
                                 mode = "train",
                                 split_ratio = self.split_ratio,
                                 image_size = self.img_size) 
        self.train_loader = DataLoader(self.train_set,
                                       batch_size = self.batch_size,
                                       shuffle = True,
                                       drop_last = True,
                                       num_workers = int(os.cpu_count() / 2))
        self.test_set = Doodle(root_path = self.data_path,
                               mode = "test",
                               split_ratio = self.split_ratio,
                               image_size = self.img_size)
        self.test_loader = DataLoader(self.test_set,
                                      batch_size = self.batch_size,
                                      shuffle = False,
                                      drop_last = False,
                                      num_workers = int(os.cpu_count() / 2))
        
    def __setUpModel(self):
        self.model = DoodleCNN(input_size = self.img_size, num_classes = len(self.train_set.classes))
        if self.resume_training:
            self.model.load_state_dict(self.saved_data["model"])
        self.model.to(self.device)
        
    def __setUpCriterion(self):
        self.criterion = nn.CrossEntropyLoss(reduction = "none")
        
    def __setUpOptimizer(self):
        self.optimizer = torch.optim.AdamW(params = self.model.parameters(), lr = self.learning_rate)
        if self.resume_training:
            self.optimizer.load_state_dict(self.saved_data["optimizer"])

    def __setUpTrainingParameters(self):
        if self.resume_training:
            self.cur_epoch = self.saved_data["cur_epoch"] + 1
            self.best_acc = self.saved_data["best_acc"]
            self.cur_patience = self.saved_data["cur_patience"]
        else:
            self.best_acc = 0    
            self.cur_epoch = 0
            self.cur_patience = 0
    
    def train(self):
        self.__setUpEnvironment()
        self.__setUpData()
        self.__setUpModel()
        self.__setUpCriterion()
        self.__setUpOptimizer()
        self.__setUpTrainingParameters()

        #training
        for epoch in range(self.cur_epoch, self.num_epochs):
            #training stage
            self.model.train()
            train_loss = 0.0
            train_bar = tqdm(self.train_loader, colour = "green")

            for iter, (images, labels) in enumerate(train_bar):
                images = images.to(self.device)
                labels = labels.to(self.device)

                #forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss_mean = loss.mean()
                loss_std = loss.std()
                mask = loss < (loss_mean + 2 * loss_std)
                if mask.sum() > 0:
                    loss = loss[mask].mean()
                    # backward and optimize as usual
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    #calculate avg loss
                    train_loss += loss.item()
                    avg_loss = train_loss/(iter + 1)
                    train_bar.set_description(f"Epoch [{epoch + 1}/{self.num_epochs}], Device:{self.device}, Loss:{avg_loss:.2f}")
                    self.writer.add_scalar("Train_loss", avg_loss, epoch * len(self.train_loader) + iter)
                
            #validation stage
            self.model.eval()
            val_loss = 0.0
            val_bar = tqdm(self.test_loader, colour = "yellow")
            y_true = []
            y_pred = []

            with torch.no_grad():
                for iter, (images, labels) in enumerate(val_bar):
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    #forward pass
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                    #calculate loss
                    val_loss += loss.mean().item()
                    avg_loss = val_loss/(iter + 1)
                    val_bar.set_description(f"Epoch [{epoch + 1}/{self.num_epochs}], Device:{self.device}, Loss:{avg_loss:.2f}")
                    self.writer.add_scalar("Val_loss", avg_loss, epoch * len(self.test_loader) + iter)

                    #store the true and predicted labels
                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(torch.argmax(outputs, dim = 1).cpu().numpy())
            # #calculate the F1 score
            # f1 = f1_score(y_true, y_pred, average = "weighted")
            # print(f"f1_score: {f1:.2f}")
            accuracy = accuracy_score(y_true, y_pred)
            print(f"accuracy: {accuracy:.2f}")
            self.writer.add_scalar("Val_accuracy", accuracy, epoch)

            #plot confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            self.plot_confusion_matrix(self.writer, cm, self.train_set.classes, epoch)

            #save the model if the validation F1 score is improved
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.cur_patience = 0
                torch.save({
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "cur_epoch": epoch,
                    "best_acc": self.best_acc,
                    "cur_patience": self.cur_patience,
                    "classes": self.train_set.classes,
                    "image_size": self.img_size
                }, os.path.join(self.checkpoint_path, "best.pt"))
            else:
                self.cur_patience += 1
                print(f"No improvements, patience: {self.cur_patience}/{self.patience}")
                

            #save the last model
            torch.save({
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "cur_epoch": epoch,
                "best_acc": self.best_acc,
                "cur_patience": self.cur_patience,
                "classes": self.train_set.classes,
                "image_size": self.img_size
            }, os.path.join(self.checkpoint_path, "last.pt"))

            if self.cur_patience >= self.patience:
                print("Early stopping due to max patience reached!")
                break

if __name__ == "__main__":
    trainer = DoodleTrainer(num_epochs = 100, 
                            batch_size = 32, 
                            learning_rate = 1e-3, 
                            img_size = 28,
                            resume_training = True, 
                            patience = 10, 
                            split_ratio = 0.8,
                            log_path = "./doodle_classification/tensorboard", 
                            checkpoint_path = "./doodle_classification/checkpoint", 
                            data_path = "./datasets/doodles")
    trainer.train()

    # print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))