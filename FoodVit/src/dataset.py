import os

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms

from utils import download_data

# seguo la struttura di un tutorial con image folder e transform
# https://www.learnpytorch.io/08_pytorch_paper_replicating/

#Versione manuale senza nessun Transfer Learning
class MakeDataloader():
    def __init__(self,opts, filename, manual=False, transform=None ,destination="src/dataset/"):
        self.opts = opts
        self.filename = filename
        self.destination = destination

        image_path=download_data(self.filename,self.destination)

        train_dir = os.path.join(image_path,"train")
        test_dir = os.path.join(image_path,"test")

        # Crea le trasformazioni
        self.simple_transform = transforms.Compose([
            transforms.Resize((opts.img_size, opts.img_size)),
            transforms.ToTensor(),
        ])

        # Crea dataloaders
        from utils import create_dataloaders
        if manual:
            self.train_dataloader, self.test_dataloader, self.class_names = create_dataloaders(
                train_dir=train_dir,
                test_dir=test_dir,
                transform=self.simple_transform,  # use manually created transforms
                batch_size=opts.batch_size,
            )
        else:
            self.train_dataloader, self.test_dataloader, self.class_names = create_dataloaders(
                train_dir=train_dir,
                test_dir=test_dir,
                transform=transform,
                batch_size=opts.batch_size,
            )


#if __name__ == "__main__":
    # da testare appena creo dei valori
