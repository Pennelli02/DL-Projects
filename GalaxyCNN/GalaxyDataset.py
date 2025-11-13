# Per questo esperimento per realizzare un modello usando pytorch useremo un dataset di galassie ottenute da
# https://astronn.readthedocs.io/en/latest/galaxy10.html

# In questo file.py ci dedicheremo a gestire il dataset, usare i dataloaders e vedere la data augmentation

# Per prima cosa dobbiamo creare una sottoclasse di `Dataset` per lavorare con
# le nostre galassie. Il modulo utils.data di pytorch mette a disposizione la
# classe base.

import torch
from torch.utils.data import Dataset, DataLoader


# Il dataset viene distribuito in formato HDF5
class GalaxyDataset(Dataset):
    def __init__(self, opts, filename, crop=True):
        import numpy as np
        import tables
        self.opts = opts
        # Tramite la libreria `pytables` possiamo aprire in lettura il file ed
        # accedere ai contenuti tramite i loro nomi. Nel caso del nostro
        # dataset, le immagini sono memorizzate nel campo `images` mentre le
        # labels sono nel campo `ans`.
        h5 = tables.File(filename)
        X = h5.root.images[:]
        #X = (h5.root.images[:].astype(np.float32) / 255.)
        y = h5.root.ans[:]

        # I dati vengono normalizzati in [0,1]
        X = (X / 255.).astype(np.float32)

        # Le immagini sono a colori e memorizzate in formato BHWC (channel last). Questo
        # ordine va bene con TensorFlow ma non con PyTorch, che vuole le immagini in
        # ordine BCWH (channel first). Quindi dobbiamo trasporre.
        self.X = torch.tensor(X.transpose(0, 3, 1, 2))

        # Inoltre nelle opzioni abbiamo il flag `crop` che se vero significa
        # ritagliare un quadrato 224x224 dalle immagini originali che sono di
        # dimensione 256x256. Il crop ha due motivi:
        #
        # - L'immagine più piccola contiene comunque l'intera galassia
        #
        # - Molte architetture hanno costanti scelte sul formato 224x224 che è
        #   quello molto popolare di ImageNet, semplificando la relazione tra le
        #   costanti che usiamo adesso e quelle che trovate in molti articoli in
        #   letteratura.

        # Per default andiamo a prendere la parte centrale dell'immagine.
        if crop:
            margin = (256 - 224) // 2
            self.X = self.X[:, :, margin:-margin, margin:-margin]

        # Infine dobbiamo costruire i vettori one-hot per i target
        # I vettori one-hot sono rappresentazioni numeriche delle classi: ogni classe è codificata
        # come un vettore di zeri con un solo 1 nella posizione corrispondente alla classe.
        # Servono per permettere al modello di trattare le categorie come indipendenti e non ordinate.

        self.num_classes = len(np.unique(y))

        EYE = np.eye(self.num_classes)
        y_oh = EYE[y]
        self.y = torch.tensor(y_oh)
        self.data_shape = X[0].shape

    # Per completare la classe dobbiamo scrivere altri due metodi:
    #
    # - `__len__`, che ritorna il numero di esempi (questo servirà per capire
    #   quanti minibatches ci sono in un'epoca)
    #
    # - `__get_item` che dato un indice `i` ritorna il dato i-esimo nel dataset.
    #   Trattandosi di semplice apprendimento supervisionato single-task,
    #   dobbiamo ritornare semplicemente una tupla (immagine, classe).

    def __len__(self):
        return self.X.shape[0]

    #
    def __getitem__(self, i):
        return self.X[i], self.y[i]

    def describe(self):
        """
        Stampa una descrizione generale del dataset in modo dinamico.
        Mostra i principali attributi tensori (es. immagini, target, ecc.)
        con nome, forma e tipo.
        """
        print("=== Dataset Info ===")
        print(f"Numero di esempi: {len(self)}")

        # Cerca tra tutti gli attributi quelli che sono tensori o array
        for attr_name, value in self.__dict__.items():
            if isinstance(value, (torch.Tensor,)):
                print(f"{attr_name:15s} → shape={tuple(value.shape)}, dtype={value.dtype}")
            elif hasattr(value, "shape"):  # per oggetti numpy
                print(f"{attr_name:15s} → shape={tuple(value.shape)}, tipo=numpy array")
