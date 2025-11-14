import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, in_channels, num_filters, kernel_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=num_filters, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

    def forward(self, x):
        return self.layers(x)


# Nel caso volessi implementare un ResNet o idea simile
class BlockRes(nn.Module):
    def __init__(self, in_channels, num_filters, kernel_size, residual=True):
        super().__init__()
        self.residual = residual
        # senza sequential
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=num_filters,
                               kernel_size=kernel_size,
                               padding='same')
        self.norm1 = nn.BatchNorm2d(num_filters)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=num_filters,
                               out_channels=num_filters,
                               kernel_size=kernel_size,
                               padding='same')
        self.norm2 = nn.BatchNorm2d(num_filters)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))

    #l'abbiamo impostato noi così non va sempre bene
    def forward(self, x):
        h = x
        h = self.conv1(h)
        h = self.norm1(h)
        h = self.relu1(h)
        h = self.conv2(h)
        if self.residual:
            h = h + x
        h = self.norm2(h)
        h = self.relu2(h)
        h = self.pool(h)
        return h


class SimpleCNN(nn.Module):
    def __init__(self, num_filters, mlp_size, num_classes):
        super().__init__()
        self.blocks = nn.Sequential(
            Block(3, num_filters, (5, 5)),
            Block(num_filters, num_filters * 2, (3, 3)),
            Block(num_filters * 2, num_filters * 4, (3, 3)),
            Block(num_filters * 4, num_filters * 8, (3, 3)),
        )
        self.bottleneck = nn.Conv2d(
            in_channels=num_filters * 8,
            out_channels=num_filters,
            kernel_size=(1, 1))
        self.flatten = nn.Flatten()  # Unroll the last feature map into a vector
        self.mlp = nn.Sequential(
            nn.Linear(in_features=14 * 14 * num_filters, out_features=mlp_size),
            nn.ReLU())
        self.head = nn.Linear(in_features=mlp_size, out_features=num_classes)

    def forward(self, x):
        h = x
        h = self.blocks(h)
        h = self.bottleneck(h)
        h = self.flatten(h)
        h = self.mlp(h)
        h = self.head(h)
        return h


# per visulizzare la struttura del modello 2 versioni usare il formato ONNX o torchvision che è sulla console
# Facciamo con torch vision

def main():
    from torchinfo import summary
    from rich.console import Console
    console = Console()
    batch_size = 4
    input_size = 224
    input_data = torch.randn(batch_size, 3, input_size, input_size)
    num_filters = 32
    num_classes = 10
    model = SimpleCNN(num_filters=num_filters,
                      mlp_size=128,
                      num_classes=num_classes)
    _ = model(input_data)
    model_stats = summary(
        model,
        input_data=input_data,
        col_names=["input_size", "output_size", "num_params"],
        row_settings=("var_names",),
        col_width=18,
        depth=8,
        verbose=0,
    )
    console.print(model_stats)


if __name__ == "__main__":
    from ipdb import launch_ipdb_on_exception

    with launch_ipdb_on_exception():
        main()
