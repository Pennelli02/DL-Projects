# Per questo esperimento per realizzare un modello usando pytorch useremo un dataset di galassie ottenute da
# https://astronn.readthedocs.io/en/latest/galaxy10.html

# In questo file.py ci dedicheremo a gestire il dataset, usare i dataloaders e vedere la data augmentation

# Per prima cosa dobbiamo creare una sottoclasse di `Dataset` per lavorare con
# le nostre galassie. Il modulo utils.data di pytorch mette a disposizione la
# classe base.

import torch
from torch.utils.data import Dataset


# Il dataset viene distribuito in formato HDF5
class GalaxyDataset(Dataset):
    def __init__(self, opts, filename, crop=True):
        import numpy as np
        import tables
        self.opts = opts
        self.crop = crop

        # NON caricare i dati in memoria, tieni solo il riferimento al file
        self.h5_file = tables.open_file(filename, mode='r')
        self.images = self.h5_file.root.images
        self.labels = self.h5_file.root.ans[:]  # Le label sono piccole, possiamo caricarle

        # Costruzione dei vettori one-hot per i target
        self.num_classes = len(np.unique(self.labels))
        EYE = np.eye(self.num_classes)
        y_oh = EYE[self.labels]
        self.y = torch.tensor(y_oh, dtype=torch.float32)

        # Memorizza la forma per riferimento
        self.data_shape = (3, 224 if crop else 256, 224 if crop else 256)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        # Carica SOLO l'immagine i-esima quando serve
        img = self.images[i]

        # Converti e normalizza
        img = img.astype('float32') / 255.0

        # Trasposizione da HWC a CHW
        img = img.transpose(2, 0, 1)

        # Crop se richiesto
        if self.crop:
            margin = (256 - 224) // 2
            img = img[:, margin:-margin, margin:-margin]

        # Converti a tensor
        img_tensor = torch.from_numpy(img)

        return img_tensor, self.y[i]

    def __del__(self):
        # Chiudi il file quando l'oggetto viene distrutto
        if hasattr(self, 'h5_file'):
            self.h5_file.close()

    def describe(self):
        """
        Stampa una descrizione generale del dataset in modo dinamico.
        """
        print("=== Dataset Info ===")
        print(f"Numero di esempi: {len(self)}")
        print(f"Forma immagini: {self.data_shape}")
        print(f"Numero classi: {self.num_classes}")
        print(f"Crop attivo: {self.crop}")
        print(f"Target shape: {tuple(self.y.shape)}, dtype={self.y.dtype}")
