"""This script is to train the doodle classification model"""
from dataset import HandDrawing
from argparse import ArgumentParser
import torch
import os
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, confusion_matrix
import shutil
import matplotlib.pyplot as plt
import numpy as np

#modify the command line arguments (Eg: python3 train.py -n 10 -b 64 -l 1e-4 -o ./tensorboard -c)
def get_args():
    parser = ArgumentParser(description = "Hand drawing classification")

    parser.add_argument("--num_epochs", "-ne", type = int, default = 100, help = "Number of epochs")
    parser.add_argument("--batch_size", "-bs", type = int, default = 32, help = "Number of images in a batch")
    parser.add_argument("--learning_rate", "-lr", type = float, default = 1e-4, help = "learning rate")
    parser.add_argument("--log_path", "-lp", type = str, default = "./doodle_classification/tensorboard", help = "place to save the tensorboard")
    parser.add_argument("--checkpoint_path", "-cp", type = str, default = "./doodle_classification/checkpoint", help = "place to save the model")
    parser.add_argument("--data_path", "-dp", type = str, default = "./datasets/doodle", help = "path to the dataset")  
    parser.add_argument("--img_size", "-is", type = int, default = 224, help = "image size (i x i) to crop")
    parser.add_argument("--resume_training", "-rt", type = bool, default = True, help = "continue training from previous epoch or not")
    parser.add_argument("--patience", "-p", type = int, default = 10, help ="Maximum number of consecutive epochs without improvements")
    parser.add_argument("--image_to_take_ratio", "-ittr", type = float, default = 0.3, help = "Ratio of images per class taken from the dataset")
    parser.add_argument("--split_ratio", "-sr", type = float, default = 0.8, help = "ratio of the data used for training")
    return parser.parse_args()

#This function is used to plot the confusion matrix in the tensorboard
def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
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

#train the model
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.resume_training == False:
        if os.path.isdir(args.log_path):
            shutil.rmtree(args.log_path)
        os.makedirs(args.log_path)

        if os.path.isdir(args.checkpoint_path):
            shutil.rmtree(args.checkpoint_path)
        os.makedirs(args.checkpoint_path)

    writer = SummaryWriter(args.log_path)
    
    #preprocess the data
    train_set = HandDrawing(root_path = args.data_path,
                            mode = "train",
                            split_ratio = args.split_ratio,
                            image_to_take_ratio= args.image_to_take_ratio)
    train_loader = DataLoader(train_set,
                              batch_size = args.batch_size,
                              shuffle = True,
                              drop_last = True,
                              num_workers = 8)
    test_set = HandDrawing(root_path = args.data_path,
                            mode = "test",
                            split_ratio = args.split_ratio,
                            image_to_take_ratio= args.image_to_take_ratio)
    test_loader = DataLoader(test_set,
                             batch_size = args.batch_size,
                             shuffle = False,
                             drop_last = False,
                             num_workers = 8)

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(in_features=512, out_features = len(train_set.classes), bias=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(params = model.parameters(), lr = args.learning_rate)
        
    if args.resume_training:
        checkpoint = os.path.join(args.checkpoint_path, "last.pt")
        saved_data = torch.load(checkpoint, weights_only = False)
        model.load_state_dict(saved_data["model"])
        optimizer.load_state_dict(saved_data["optimizer"])
        cur_epoch = saved_data["cur_epoch"] + 1
        best_f1 = saved_data["best_f1"]
        cur_patience = saved_data["cur_patience"]
    else:
        best_f1 = 0    #to store the best model, for deployment
        cur_epoch = 0
        cur_patience = 0

    for epoch in range(cur_epoch, args.num_epochs):
        #training stage
        total_loss = 0
        prrogress_bar = tqdm(train_loader, colour = "green")
        model.train()
        for iter, (images, labels) in enumerate(prrogress_bar):
            #forward pass
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)

            #calculate loss
            loss = criterion(output, labels)
            total_loss += loss.item()
            avg_loss = total_loss/(iter + 1)
            prrogress_bar.set_description(f"Epoch:{epoch + 1}/{args.num_epochs}, Loss:{avg_loss:.2f}, Device:{device}")
            writer.add_scalar(tag = "Train/Loss",
                              scalar_value = avg_loss, 
                              global_step = epoch * len(train_loader) + iter)

            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        #validtion
        model.eval()
        total_loss = 0
        progress_bar = tqdm(test_loader, colour = "yellow")
        y_true = []
        y_pred = []
        with torch.no_grad():
            for iter, (images, labels) in enumerate(progress_bar):
                #forward pass
                images = images.to(device)
                labels = labels.to(device)
                output = model(images)

                #calculate loss
                total_loss += criterion(output, labels).item()
                prediction = torch.argmax(output, dim = 1)
                y_true.extend(labels.tolist())
                y_pred.extend(prediction.tolist())
        
        avg_loss = total_loss/len(test_loader)
        f1 = f1_score(y_true, y_pred, average = "weighted")
        print(f"Epoch:{epoch + 1}/{args.num_epochs}, Loss:{avg_loss:.2f}, f1_score:{f1:.2f}")
        writer.add_scalar(tag = "Eval/Loss",
                              scalar_value = avg_loss, 
                              global_step = epoch)
        writer.add_scalar(tag = "Eval/f1_score",
                              scalar_value = f1, 
                              global_step = epoch)
        plot_confusion_matrix(writer, confusion_matrix(y_true, y_pred), train_set.classes, epoch)
        
        #save the model
        if f1 > best_f1:
            cur_patience = 0
            best_f1 = f1
            saved_data = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "cur_epoch": epoch,
                "cur_patience": 0,
                "best_f1": f1,
                "classes": train_set.classes
            }
            checkpoint  = os.path.join(args.checkpoint_path, "best.pt")
            torch.save(saved_data, checkpoint)
        else:
            cur_patience +=1
            print(f"No improvements. Patience: {cur_patience}/{args.patience}")
        saved_data = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "cur_epoch": epoch,
            "cur_patience": cur_patience,
            "best_f1": f1,
            "categories": train_set.classes
        }
        checkpoint  = os.path.join(args.checkpoint_path, "last.pt")
        torch.save(saved_data, checkpoint)

        if cur_patience >= args.patience:
            print("Early stopping due to max patience reached!")
            break


if __name__ == "__main__":
    args = get_args()
    train(args)