import math

import torch
import torchvision
from torch import nn

"""
    esiste una versione già impostata unet-pytorch 0.4.3, noi in questo progetto la ricreremo seguendo l'articolo
    
"""
import torch.nn.init as init

# richiesta nel paper
def init_weights(m):
    """
    Applica l'inizializzazione Kaiming Normale a nn.Conv2d.
    """
    if isinstance(m, nn.Conv2d):
        # Applica Kaiming Normale (He Initialization) per la nonlinearity 'relu'
        init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

        # Inizializza i bias a zero (pratica comune)
        if m.bias is not None:
            init.constant_(m.bias, 0)

    # ConvTranspose2d (Up-convolution) è spesso trattato come un Conv2d
    # per l'inizializzazione, quindi applichiamo lo stesso schema.
    if isinstance(m, nn.ConvTranspose2d):
        init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)

class Block(nn.Module):  # non è presente una batchNorm in questo articolo quindi non verrà inserita
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3)  # padding = 0 paper ed è già settato da pytorch
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x


# facciamolo a mano con valori già settati per i canali (no cicli)
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Contraction Blocks (due conv 3x3 + ReLU)
        self.block1 = Block(1, 64)
        self.block2 = Block(64, 128)
        self.block3 = Block(128, 256)
        self.block4 = Block(256, 512)
        # 2. Bottom Block (nessun pooling dopo)
        self.block5 = Block(512, 1024)

        # 3. Pooling Layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # stride 2 presente nel paper
        # Usare una singola istanza nn.MaxPool2d e riutilizzarla è comune

    def forward(self, x):
        flrs = []

        x1 = self.block1(x)
        flrs.append(x1)
        x1 = self.pool(x1)

        x2 = self.block2(x1)
        flrs.append(x2)
        x2 = self.pool(x2)

        x3 = self.block3(x2)
        flrs.append(x3)
        x3 = self.pool(x3)

        x4 = self.block4(x3)
        flrs.append(x4)
        x4 = self.pool(x4)

        x5 = self.block5(x4)

        # Restituisce le feature (x1..x4) e il collo di bottiglia (x5)
        # Le feature devono essere concatenate nel decoder
        return x5, flrs[::-1]  # Restituisce x5 e le feature in ordine inverso (per concatenazione)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        """
        Note that in PyTorch, the ConvTranspose2d operation performs the “up-convolution”. It accepts parameters like 
        in_channels, out_channels, kernel_size and stride amongst others.
        """
        #supponendo che il blocco 5 restituisca (512-->1024)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        # ricorda che tra uno e l'altro c'è la concatenazione e il crop
        self.block1 = Block(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.block2 = Block(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.block3 = Block(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.block4 = Block(128, 64)

    def crop(self, enc_ftrs, x):
        """Implementazione della funzione di crop usando torchvision CenterCrop."""
        _, _, H, W = x.shape
        # Il CenterCrop è più sicuro del semplice slicing per ritagliare simmetricamente.
        return torchvision.transforms.CenterCrop([H, W])(enc_ftrs)


    def forward(self, x, en_filter):
        # en_filter =[x4, x3, x2, x1]
        x = self.up1(x)  #[1024-->512]
        enc = self.crop(en_filter[0], x)  #[1024-->512]
        x = torch.cat((x, enc), dim=1)  #[512-->1024]
        x = self.block1(x)  #[1024-->512]
        x = self.up2(x)
        enc = self.crop(en_filter[1], x)
        x = torch.cat((x, enc), dim=1)
        x = self.block2(x)
        x = self.up3(x)
        enc = self.crop(en_filter[2], x)
        x = torch.cat((x, enc), dim=1)
        x = self.block3(x)
        x = self.up4(x)
        enc = self.crop(en_filter[3], x)
        x = torch.cat((x, enc), dim=1)
        x = self.block4(x)
        return x


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        # Strato Finale (Segmentation Head):
        # Mappa i 64 canali di feature prodotti dall'ultimo blocco del Decoder
        # al numero di classi desiderato (2 classi: Membrana e Cella/Sfondo).
        # Un kernel 1x1 è usato per eseguire una combinazione lineare pixel-wise.
        # NON è necessaria una "Classification Head" (come i Fully Connected Layers)
        # perché la segmentazione avviene a livello di pixel
        self.conv1 = nn.Conv2d(64, 2, kernel_size=1)
        self.apply(init_weights)

    def forward(self, x):
        x_bottleneck, encoder_features = self.encoder(x) # Input: 1 canale
        dec = self.decoder(x_bottleneck, encoder_features) # Output: 64 canali
        out=self.conv1(dec) # Output: 2 canali (classi)
        # L'output sono i "logits" (attivazioni pre-Softmax) per ogni pixel
        # La funzione Softmax e la Perdita Ponderata (Weighted Cross Entropy)
        # VENGONO APPLICATE NELLA FASE DI TRAINING (funzione di Loss) e non qui.
        return out

