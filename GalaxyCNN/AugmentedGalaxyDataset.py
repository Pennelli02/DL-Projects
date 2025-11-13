# Qui applichiamo il concetto di Augmented Data che ci permette osservando il dominio dei dati di apporre delle
# modifiche e permettere di aumentare le dimensioni del dataset
from GalaxyDataset import GalaxyDataset
import torch
from torch.utils.data import Dataset, DataLoader


# Per modificare il dataloader ci basta ereditare dalla classe `GalaxyDataset`

# Nel modulo `transforms` di `torchvision` sono disponibili vari algoritmi di
# trasformazione su immagini che possiamo semplicemente includere in una pipeline
# di preprocessing realizzata dalla classe `Compose`. Le trasformazioni vengono
# applicate in sequenza. A fini dimostrativi creiamo nel costruttore l'oggetto
# `augmentation_pipeline` che mette in cascata rotazioni, crop e riflessioni casuali. Nel caso
# delle galassie, qualsiasi rotazione è accettabile quindi in
# `opts.rotation_degrees` possiamo usare tutti gli angoli tra 0° e 180°. Più in
# generale, le trasformazioni hanno iperparametri che dovrebbero essere
# ottimizzati con un validation set se la background knowledge non è sufficiente.

# Infine sovraccarichiamo la `__get_item__` per invocare la pipeline di
# augmentation.

class AugmentedGalaxyDataset(GalaxyDataset):

    def __init__(self, opts, filename):
        from torchvision.transforms import v2
        super().__init__(opts, filename, crop=False)
        self.augmentation_pipeline = v2.Compose([
            v2.RandomRotation(degrees=self.opts.rotation_degrees, expand=False),
            v2.RandomCrop(size=224),
            v2.RandomHorizontalFlip(p=self.opts.horizontal_flip_probability),
            v2.RandomVerticalFlip(p=self.opts.vertical_flip_probability),
        ])

    def __getitem__(self, i):
        return self.augmentation_pipeline(self.X[i]), self.y[i]


# Adesso passiamo a costruire i dataloaders. Questo è molto semplice: basta
# creare un oggetto di classe `DataLoader` a partire da un oggetto di classe
# `Dataset`. In questo esempio creiamo due datasets uno di train e uno di test.
# Lo split non viene stratificato (ovvero non è garantito che le proporzioni
# delle diverse classi sia conservata in ciascun split). Questo non è problematico per datasets abbastanza grandi.

# Usiamo i parametri di default
class MakeDataLoaders:
    def __init__(self, opts, data):
        generator = torch.Generator().manual_seed(opts.seed)
        train, test = torch.utils.data.random_split(data, lengths=[1 - opts.test_size, opts.test_size],
                                                    generator=generator)
        self.train_dataloader = DataLoader(train, batch_size=opts.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test, batch_size=opts.batch_size, shuffle=True)
