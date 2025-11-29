"""
    Per praticità creremo una versione di Vit, però come spiegato nell'articolo si userà una versione pre trained con imageNet
    ci concentriamo sulla versione di Vit base
"""

import torch
import torchvision
from torch import nn
from torchvision import models

# versione NOT PRETRAINED

class PatchEmbedder(nn.Module):
    """Turns a 2D input image into a 1D sequence learnable embedding vector.

        Args:
            in_channels (int): Number of color channels for the input images. Defaults to 3.
            patch_size (int): Size of patches to convert input image into. Defaults to 16.
            embedding_dim (int): Size of embedding to turn image into. Defaults to 768. (dimensione nascosta D)
    """
    def __init__(self, in_channels: int = 3, patch_size: int = 16, embedding_dim: int = 768, *args, **kwargs):
        super().__init__()

        # Create a layer to turn an image into patches
        self.patcher = nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size)
        # Flatten(start_dim=2, end_dim=3) appiattisce SOLO le ultime due dimensioni del tensore.
        # È usato nel Vision Transformer per convertire ogni patch da una mini-immagine 2D (P x P)
        # a un vettore 1D di lunghezza P*P.
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patcher(x)
        x = self.flatten(x)
        return x.permute(0, 2, 1)
        #Permute(0, 2, 1) riordina le dimensioni del tensore.
        # Dopo il flatten dei patch, il tensore ha forma:
        #   [batch_size, P^2•C, num_patches]
        # dove P^2•C è la dimensione di ogni patch flattenato.
        #
        # Tuttavia, il Vision Transformer si aspetta i token (patch) come seconda dimensione:
        #   [batch_size, num_patches, patch_dimension]
        #
        # Quindi permutiamo gli assi 1 e 2:
        #   da [B, P^2•C, N]
        #   a  [B, N, P^2•C]
        #
        # In questo modo ogni patch diventa un "token" e può essere passato correttamente
        # al Transformer Encoder (che lavora sulla dimensione N dei token).

class MultiheadSelfAttentionBlock(nn.Module):
    # Creates a multi-head self-attention block
    # valori presente nel paper non risulti qui sia presente un dropout
    def __init__(self, embed_dim: int = 768, num_heads: int= 12, dropout: float = 0):
        super().__init__()

        self.norm1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        x = self.norm1(x)
        # Self-attention layer:
        # - query, key e value sono tutti 'x' perché siamo in self-attention:
        #   ogni token (patch) dell'immagine confronta sé stesso con tutti gli altri token.
        # - multihead_attn ritorna:
        #       attn_output: i token aggiornati dopo il meccanismo di attenzione
        #       attn_weights: le matrici di attenzione (non calcolate qui)
        # - need_weights=False evita di calcolare e restituire le attention maps,
        #   risparmiando memoria e tempo di computazione.
        attn_output, _ = self.attn(query=x, key=x, value=x, need_weights=False)
        return attn_output

class MLPBlock(nn.Module):
    # The MLP contains two layers with a GELU non-linearity
    # Dropout, when used, is applied after every dense layer except for the qkv-projections and directly after adding
    # positional- to patch embeddings
    # layer norm -> linear layer -> non-linear layer -> dropout -> linear layer -> dropout
    # paper dropout= 0.1
    def __init__(self, embed_dim: int = 768, mlp_size= 3072, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(normalized_shape=embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_size, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x)
        x = self.mlp(x)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim: int = 768, num_heads: int= 12, mlp_size= 3072, dropout: float = 0.1, attn_dropout: float = 0):
        super().__init__()

        self.msa_block = MultiheadSelfAttentionBlock(embed_dim, num_heads, dropout=attn_dropout)

        self.mlp_block = MLPBlock(embed_dim, mlp_size, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Create residual connection for MSA block (add the input to the output)
        x = self.msa_block(x) + x

        # Create residual connection for MLP block (add the input to the output)
        x = self.mlp_block(x) + x

        return x


# qui creiamo e gestiamo la position patch embedding e dal paper risulta
# gli embedding sono presenti dei dropuot layer
class ViT(nn.Module):
    # Vit model
    def __init__(self, num_classes:int, img_size:int=224, in_channels:int=3, patch_size:int=16, n_trs_layers:int=12, embedding_dim:int=768, mlp_size:int=3072, heads:int=12, mlp_dropout:float=0.1, attn_dropout:float=0, embedding_dropout:float=0.1):
        super().__init__()

        #Create learnable class embedding
        self.class_embedding= nn.Parameter(torch.rand(1, 1, embedding_dim), requires_grad=True)

        self.num_patches = (img_size * img_size) // patch_size ** 2

        #Create learnable position embedding
        self.position_embedding = nn.Parameter(torch.rand(1, self.num_patches+1, embedding_dim), requires_grad=True)

        #Create dropout embedding
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        #Create patch embedding layer
        self.patcher = PatchEmbedder(in_channels=in_channels, patch_size=patch_size, embedding_dim=embedding_dim)

        self.encoder_layer = nn.Sequential(*[TransformerEncoderBlock(embedding_dim, heads, mlp_size, mlp_dropout, attn_dropout) for _ in range(n_trs_layers)])

        #Classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(embedding_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        batch_size = x.shape[0]

        #Create class token embedding and expand it to match the batch size
        class_token = self.class_embedding.expand(batch_size, -1, -1) ## "-1" means to infer the dimension

        #Create patch embedding
        x = self.patcher(x)

        #Concat class embedding and patch embedding
        x = torch.cat((class_token, x), dim=1)

        #Add position embedding to patch embedding
        x = self.position_embedding + x

        x = self.embedding_dropout(x)

        x = self.encoder_layer(x)

        #Put 0 index logit through classifier
        x = self.classifier(x[:, 0]) # run on each sample in a batch at 0 index

        return x


# versione PRETRAINED
class PretrainedViT(nn.Module):
    """Vision Transformer ViT-Base/16 pre-trained su ImageNet.

    Args:
        num_classes: Numero di classi del tuo dataset
    """

    def __init__(self, num_classes: int = 3):
        super().__init__()

        # Carica ViT-B/16 pre-trained
        weights = torchvision.models.ViT_B_16_Weights.DEFAULT
        self.model = torchvision.models.vit_b_16(weights=weights)

        # Modifica classification head
        # Il ViT ha: model.heads.head (Linear layer)
        in_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(
            in_features=in_features,
            out_features=num_classes
        )

        for name, p in self.model.named_parameters():
            if "heads" not in name:  # train only classification head
                p.requires_grad = False

    def forward(self, x):
        return self.model(x)

    def get_transforms(self):
        """Ritorna le trasformazioni richieste per il modello pre-trained."""
        weights = torchvision.models.ViT_B_16_Weights.DEFAULT
        return weights.transforms()