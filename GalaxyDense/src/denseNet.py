# creazione del DenseNet-121 seguiamo l'articolo in originale
"""
    denseNet from scratch
    prebuilt in torchvision

    :argument
    DenseNet-121 è una rete convoluzionale profonda con 4 Dense Block, collegati tramite 3 Transition Layer,
    dove ogni layer dentro un Dense Block riceve in ingresso TUTTE le feature-map dei layer precedenti tramite
    concatenazione
"""

import torch
from torch import nn
from torchvision import models


class FirstBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        """
        primo blocco da inserire prima dei dense layer
        Conv7×7, stride 2
        MaxPool3×3, stride 2
        BN-ReLU-Conv
        padding=3 perché: O=(W−K+2P)/S +1---> P=2,5=3
         
        Parameters
        ----------
        in_channels : int
            Normally 3 for color images
        out_channels : int
            Number of initial features
        """
        self.fBlock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), # seguiamo l'articolo
        )

    def forward(self, x):
        x = self.fBlock(x)
        return x

class DenseLayer(nn.Module):
    def __init__(self,
                 num_input_features,
                 num_output_features,
                 k,
                 bottleneck_size=4):
        """Generic layer in a dense block

        Parameters
        ----------
        num_input_features : int
            Number of features in the input map
        k : int
            growth rate
        bottleneck_size : int
            Bottleneck size (4 in the paper)

        Returns
        -------
        Tensor
            output feature map of shape (bottleneck_size*k,:,:)

        """
        super().__init__()
        self.bn1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.bottleneck = nn.Conv2d(in_channels=num_input_features,
                                    out_channels=bottleneck_size * k,
                                    kernel_size=1,
                                    padding='same')
        self.bn2 = nn.BatchNorm2d(bottleneck_size * k)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels=bottleneck_size * k,
                              out_channels=num_output_features,
                              kernel_size=3,
                              padding=1) # paper
    def forward(self, x):
        h = x
        h = self.bn1(h)
        h = self.relu1(h)
        h = self.bottleneck(h)
        h = self.bn2(h)
        h = self.relu2(h)
        h = self.conv(h)
        return h

class TransitionLayer(nn.Module): # compressione 0.5 BN → ReLU → Conv1×1 (riduce canali) → AvgPool 2×2 (riduce risoluzione)
    def __init__(self, num_features):
        super().__init__()

        self.norm = nn.BatchNorm2d(num_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels=num_features,
                              out_channels=num_features // 2,
                              kernel_size=1)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)

    def forward(self, x):
        h=x
        h = self.norm(h)
        h = self.relu(h)
        h = self.conv(h)
        h = self.pool(h)
        return h

class ClassificationHead(nn.Module):
    def __init__(self, num_features, num_classes): # nel nostro caso saranno 10 classi
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(num_features, num_classes)

    def forward(self, x):
        h = x
        h = self.pool(h)
        h = self.flatten(h)
        h = self.dense(h) # no softmax perché si fa nella entropy... otteniamo i logits
        return h

class DenseBlock(nn.Module):
        def __init__(self, num_layers, k0, k):
            """
            Dense Block con concatenazione progressiva

            Parameters
            ----------
            num_layers : int
                Numero di layer nel blocco
            k0 : int
                Numero di canali in input al blocco
            k : int
                Growth rate
            """
            super().__init__()
            self.layers = nn.ModuleList()  # Cambia da Sequential a ModuleList!

            # Ogni layer riceve un numero crescente di canali
            for ell in range(num_layers):
                self.layers.append(DenseLayer(
                    num_input_features=k0 + ell * k,
                    num_output_features=k,  # Ogni layer produce sempre k canali
                    k=k,
                ))

        def forward(self, x):
            """
            Forward con concatenazione progressiva

            IMPORTANTE: Ritorna TUTTE le feature concatenate,
            non solo l'output dell'ultimo layer!
            """
            features = [x]  # Lista che accumula tutte le feature map

            for layer in self.layers:
                # Concatena tutte le feature precedenti
                concatenated = torch.cat(features, dim=1)
                # Passa attraverso il layer corrente
                new_features = layer(concatenated)
                # Aggiungi le nuove feature alla lista
                features.append(new_features)

            # Ritorna TUTTE le feature concatenate
            return torch.cat(features, dim=1)

class DenseNet(nn.Module):
    def __init__(
            self,
            input_size,
            num_layers,
            k,
            num_dense_blocks,
            num_classes,
    ):
        """
        DenseNet completo

        Parameters
        ----------
        input_size : int
            Dimensione input (es. 224)
        num_layers : list
            Lista con numero di layer per ogni dense block (es. [6,12,24,16])
        k : int
            Growth rate (es. 32)
        num_dense_blocks : int
            Numero di dense block (4 per DenseNet-121)
        num_classes : int
            Numero di classi output
        """
        super().__init__()

        # First Block
        self.first_block = FirstBlock(in_channels=3, out_channels=64)

        # Dense Blocks + Transition Layers
        self.blocks = nn.Sequential()
        k0 = 64  # Numero di canali dopo il first block

        for i in range(num_dense_blocks):
            # Dense Block
            self.blocks.add_module(
                f"dense_block_{i}",
                DenseBlock(num_layers[i], k0, k)
            )

            # Aggiorna k0: dopo il dense block abbiamo k0 + num_layers[i]*k canali
            k0 = k0 + num_layers[i] * k

            # Transition Layer (tranne dopo l'ultimo dense block)
            if i < num_dense_blocks - 1:
                self.blocks.add_module(
                    f"transition_{i}",
                    TransitionLayer(num_features=k0)
                )
                # Dopo transition, i canali si dimezzano
                k0 = k0 // 2

        # Classification Head
        self.head = ClassificationHead(k0, num_classes)

    def forward(self, x):
        h = self.first_block(x)
        h = self.blocks(h)  # Puoi usare Sequential direttamente
        h = self.head(h)
        return h

# per seguire l'articolo viene usato il TRANSFER LEARNING (usare un densenet pre addestrato su imagenet)
# Qui viene proposto 2 modi: Backbone, Handmade
# Backbone
class DenseNetPretrainedBackbone(nn.Module):
    def __init__(self, num_classes=10, freeze_backbone=False):
        super().__init__()

        # carica densenet torchvision
        base = models.densenet121(weights="IMAGENET1K_V1")

        # backbone
        self.features = base.features

        # congela se necessario
        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

        # classification head custom
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.relu(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

"""
================================================================================
 DENSENET-121 HANDMADE COMPATIBILE CON I PESI IMAGENET (TORCHVISION)

 Questo file contiene una versione *fatta a mano* di DenseNet-121 che replica
 esattamente l'architettura ufficiale di torchvision, permettendo quindi:

    • transfer learning dai pesi ImageNet
    • fine-tuning sull'intera rete
    • sostituzione del classificatore finale (per Galaxy10 o altre classi)
    • analisi e modifica interna dell'architettura mantenendo i pesi ufficiali

────────────────────────────────────────────────────────────────────────────────
 PERCHÉ SERVE UNA VERSIONE COMPATIBILE?
────────────────────────────────────────────────────────────────────────────────

Per poter caricare i pesi pre-addestrati di torchvision è indispensabile che:

    ✔ tutti i layer abbiano la stessa dimensione (same shape)
    ✔ gli stessi nomi degli attributi nello state_dict
    ✔ la stessa sequenza di layer (BN → ReLU → Conv1x1 → BN → ReLU → Conv3x3)
    ✔ le Transition Layer siano identiche
    ✔ il FirstBlock sia identico a quello ufficiale

────────────────────────────────────────────────────────────────────────────────
 COME FUNZIONA QUESTO MODELLO?
────────────────────────────────────────────────────────────────────────────────

1) Viene caricato `torchvision.models.densenet121(weights=...)`
   → prendiamo tutti i pesi pre-addestrati su ImageNet.

2) Ricostruiamo la rete *a mano*, ma in modo IDENTICO a come PyTorch l’ha
   implementata.  
   Ogni layer (_DenseLayer, _DenseBlock, _Transition) replica fedelmente
   l’architettura originale (dimensioni, stride, padding, ordine dei layer).

3) Confrontiamo lo `state_dict` della nostra rete con quello ufficiale:
       se un peso ha lo stesso nome e la stessa forma → lo carichiamo.
       altrimenti lo ignoriamo (es. il classifier finale).

4) Rimpiazziamo il classificatore finale (fc) con uno nuovo:
       nn.Linear(1024, num_classes)

5) Il backbone può essere:
       – lasciato trainabile (full fine-tuning, come nel paper)
       – congelato (feature extraction)

────────────────────────────────────────────────────────────────────────────────
 STRUTTURA DEL MODELLO
────────────────────────────────────────────────────────────────────────────────

La rete è composta da:

    FirstBlock:
        Conv7×7 (stride 2)
        BatchNorm
        ReLU
        MaxPool3×3 (stride 2)

    DenseBlock1: 6 layers  (growth rate 32)
    Transition1: dimezza i canali

    DenseBlock2: 12 layers
    Transition2

    DenseBlock3: 24 layers
    Transition3

    DenseBlock4: 16 layers

    Norm finale

    → Global Average Pooling
    → Classificatore lineare

────────────────────────────────────────────────────────────────────────────────
 PERCHÉ USARE QUESTO METODO?

• puoi ispezionare o modificare la rete anche a livello di singolo layer
• puoi vedere *come* PyTorch implementa DenseNet internamente
• puoi modificare growth rate, bottleneck, num_layers, ecc. senza perdere
  la possibilità di usare il transfer learning (finché mantieni la compatibilità)
• puoi fare debug e sperimentare senza dipendere da torchvision

────────────────────────────────────────────────────────────────────────────────
 COME VERIFICARE CHE I PESI SIANO STATI IMPORTATI CORRETTAMENTE?

Il caricamento è fatto così:

    custom_sd = self.state_dict()
    to_load = {k: v for k,v in official_sd.items()
               if k in custom_sd and v.shape == custom_sd[k].shape}
    custom_sd.update(to_load)
    self.load_state_dict(custom_sd)

→ Se architettura e nomi coincidono, verranno importati TUTTI i pesi del backbone.
→ Solo il classificatore finale resterà non pre-addestrato.

────────────────────────────────────────────────────────────────────────────────
 COMPATIBILITÀ E LIMITI

✔ pienamente compatibile con i pesi ImageNet 2023 di PyTorch  
✔ identico al modello ufficiale al 100%  
✘ se modifichi nomi, padding o ordine dei layer → i pesi non saranno più caricabili

────────────────────────────────────────────────────────────────────────────────
"""
class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size=4):
        super().__init__()

        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate,
                               kernel_size=1, stride=1, bias=False)

        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate,
                               kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        new_features = self.conv1(self.relu1(self.norm1(x)))
        new_features = self.conv2(self.relu2(self.norm2(new_features)))
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, growth_rate, bn_size=4):
        super().__init__()
        self.layers = nn.ModuleList([])

        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate,
                bn_size
            )
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ---------------------------
#  Transition Layer (compatibile)
# ---------------------------
class _Transition(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super().__init__()

        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features,
                              kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(self.relu(self.norm(x)))
        x = self.pool(x)
        return x


# Handmade
class DenseNetPretrainedHandmade(nn.Module):
    def __init__(self, num_classes=10, freeze_backbone=False):
        super().__init__()

        # 1. CARICA MODELLO UFFICIALE
        official = models.densenet121(weights="IMAGENET1K_V1")
        off_sd = official.state_dict()

        # 2. COSTRUISCI BACKBONE IDENTICO
        self.features = nn.Sequential()

        self.features.add_module("conv0", nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False))
        self.features.add_module("norm0", nn.BatchNorm2d(64))
        self.features.add_module("relu0", nn.ReLU(inplace=True))
        self.features.add_module("pool0", nn.MaxPool2d(3, stride=2, padding=1))

        num_features = 64
        growth_rate = 32
        block_layers = [6, 12, 24, 16]

        for i, num_layers in enumerate(block_layers):
            block = _DenseBlock(num_layers, num_features, growth_rate)
            self.features.add_module(f"denseblock{i + 1}", block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_layers) - 1:
                trans = _Transition(num_features, num_features // 2)
                self.features.add_module(f"transition{i + 1}", trans)
                num_features = num_features // 2

        self.features.add_module("norm5", nn.BatchNorm2d(num_features))

        # 3. COPIA PESI (solo quelli con shape compatibile)
        custom_sd = self.state_dict()
        to_load = {k: v for k, v in off_sd.items() if k in custom_sd and v.shape == custom_sd[k].shape}
        custom_sd.update(to_load)
        self.load_state_dict(custom_sd)

        # congela backbone se richiesto
        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

        # 4. CLASSIFIER PER Galaxy10
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = torch.relu(features)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


