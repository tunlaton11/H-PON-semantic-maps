from dataset import NuSceneDataset
from model import UNET

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader

import platform
import re


def main():
    train_dataset = NuSceneDataset(data_root="data", label_dir="labels")

    train_loader = DataLoader(train_dataset,
                            batch_size=2,
                            num_workers=2,
                            pin_memory=True,
                            shuffle=True)
    

    network = UNET(in_channels=3, out_channels=14)
    
    this_device = platform.platform()
    # if torch.cuda.is_available():
    #     device = "cuda:0"
    # elif re.search("arm64", this_device):
    #     # use Apple GPU
    #     device = "mps"
    # else:
    #     device = "cpu"
    device = "cpu"

    loss_fn = nn.CrossEntropyLoss()
    # loss_fn = nn.BCELoss()
    optimizer = optim.Adam(network.parameters(), lr=0.001)

    network.to(device)


    for epoch in range(20):
        print(f'Training epoch {epoch+1}...')
        for batch_idx, batch in enumerate(train_loader):
            
            image, labels, mask = batch
            image = image.to(device)
            labels = labels.to(device).type(torch.FloatTensor)
            mask = mask.to(device)
            
            prediction = network(image).to(device)

            # print('pred', prediction.shape, type(prediction))
            # print('true label', labels.shape, type(labels))

            # 4.2 compute loss
            loss = loss_fn(prediction, labels).to(device)

            # 4.3 compute gradient
            optimizer.zero_grad()
            loss.backward()

            # 4.4 update weights
            optimizer.step()
            print('loss ', loss.item())



if __name__ == "__main__":
    main()