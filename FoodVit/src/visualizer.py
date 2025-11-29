import matplotlib.pyplot as plt
from torchinfo import summary
from rich.console import Console
console = Console()

def visualize(model, model_name, input_data):
    out = model(input_data)
    console.print(f'Computed output, shape = {out.shape=}')
    model_stats = summary(model,
                          input_data=input_data,
                          col_names=[
                              "input_size",
                              "output_size",
                              "num_params",
                              # "params_percent",
                              # "kernel_size",
                              # "mult_adds",
                          ],
                          row_settings=("var_names",),
                          col_width=18,
                          depth=8,
                          verbose=0,
                          )
    console.print(model_stats)

def visualize_imported_images(train_dl, quantity, class_names):
    # Get a batch of images
    image_batch, label_batch = next(iter(train_dl))

    # Get a single image from the batch
    image, label = image_batch[quantity-1], label_batch[quantity-1]

    # View the batch shapes
    print(image.shape, label)
    # Plot image with matplotlib
    plt.imshow(image.permute(1, 2,
                             0))  # rearrange image dimensions to suit matplotlib [color_channels, height, width] -> [height, width, color_channels]
    plt.title(class_names[label])
    plt.axis(False)

