import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class GalaxyDataset(Dataset):

    def __init__(self, opts, filename):

        import tables
        self.opts = opts

        h5 = tables.File(filename)
        X = h5.root.images[:]
        y = h5.root.ans[:]

        # APPLICA MORPHOLOGICAL OPENING PRIMA DELLA NORMALIZZAZIONE (richiesta nell'articolo)
        if opts.use_morphological_opening:
            X = self.apply_morphological_opening(X, kernel_size=opts.morph_kernel_size)

        #articolo normalizza tra [-1,1]
        X = (X / 127.5 - 1).astype(np.float32)

        self.X = torch.tensor(X.transpose(0, 3, 1, 2))

        # Infine dobbiamo costruire i vettori one-hot per i target
        self.num_classes = len(np.unique(y))

        # Non si usa più l'One-Hot Encoding per i target.
        # CrossEntropyLoss si aspetta gli indici di classe (Label Encoding)
        # di tipo torch.long per poter applicare i class_weights.
        # =========================================================================

        # self.num_classes = len(np.unique(y)) # Manteniamo per coerenza
        # EYE = np.eye(self.num_classes)
        # y_oh = EYE[y]
        # self.y = torch.tensor(y_oh) # Rimosso l'One-Hot Encoding

        self.num_classes = len(np.unique(y))
        self.y = torch.tensor(y, dtype=torch.long) # Target come indici interi (Label Encoding)

        self.data_shape = X[0].shape

    def __len__(self):
        return self.X.shape[0]

    #
    def __getitem__(self, i):
        return self.X[i], self.y[i]

    def apply_morphological_opening(self, images, kernel_size=5, kernel_shape='ellipse'):
        import cv2

        # Crea il kernel (strutturing element) in base al tipo richiesto
        if kernel_shape == 'ellipse':
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (kernel_size, kernel_size))
        elif kernel_shape == 'rect':
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                               (kernel_size, kernel_size))

        processed = []

        # Cicla su ogni immagine della lista
        for img in images:

            # Se l'immagine ha 3 canali (RGB)
            if img.shape[-1] == 3:

                # Applica l'opening a ciascun canale separatamente
                channels = [
                    cv2.morphologyEx(
                        img[:, :, c].astype(np.uint8),  # singolo canale
                        cv2.MORPH_OPEN,  # tipo di operazione morfologica
                        kernel  # kernel scelto
                    )
                    for c in range(3)
                ]

                # Ricompone l'immagine combinando i tre canali
                processed.append(np.stack(channels, axis=-1))

            else:
                # Immagine in scala di grigi: operazione diretta
                processed.append(
                    cv2.morphologyEx(img.astype(np.uint8),
                                     cv2.MORPH_OPEN,
                                     kernel)
                )

        # Restituisce la lista convertita in array numpy
        return np.array(processed)

# IN TEORIA NELL'ARTICOLO SI DEVE SPOSTARE IN GPU E NORMALIZZAZIONE FATTA DAL MODELLO
class AugmentedGalaxyDataset(GalaxyDataset):

    def __init__(self, opts, filename):
        from torchvision.transforms import v2
        super().__init__(opts, filename)
        self.augmentation_pipeline = v2.Compose([
            v2.RandomRotation(degrees=self.opts.rotation_degrees, expand=False),
            v2.CenterCrop(size=224),
            v2.RandomAffine(
                degrees=[0, 0],  # Mantieni rotazione zero, poiché usi RandomRotation separatamente.
                translate=(self.opts.translate_x, self.opts.translate_y),
                fill=0
            ),
            v2.RandomAutocontrast(),
            v2.RandomHorizontalFlip(p=self.opts.horizontal_flip_probability),
            v2.RandomVerticalFlip(p=self.opts.vertical_flip_probability),
        ])

    def __getitem__(self, i):
        return self.augmentation_pipeline(self.X[i]), self.y[i]

# nell'articolo usano 70:15:15 rispettivamente per training validation e test
class MakeDataLoaders():
    def __init__(self, opts, data):
        generator = torch.Generator().manual_seed(opts.seed)
        # Calcola le dimensioni dei 3 split
        total_size = len(data)
        train_size = int(0.70 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size  # Per evitare errori di arrotondamento

        # Split in 3 parti
        train, validation, test = torch.utils.data.random_split(
            data,
            lengths=[train_size, val_size, test_size],
            generator=generator
        )

        # Crea i 3 DataLoader
        self.train_dataloader = DataLoader(
            train,
            batch_size=opts.batch_size,
            shuffle=True
        )
        self.validation_dataloader = DataLoader(
            validation,
            batch_size=opts.batch_size,
            shuffle=False  # Validation non va shufflato
        )
        self.test_dataloader = DataLoader(
            test,
            batch_size=opts.batch_size,
            shuffle=False  # Test non va shufflato
        )

