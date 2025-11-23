import torch


# def dice_coefficient(pred, target, smooth=1e-6):
#     """Calcola Dice coefficient per segmentazione binaria"""
#     pred = torch.sigmoid(pred)  # se usi BCE with logits
#     pred = (pred > 0.5).float()
#
#     # Questo trasforma [B, 1, H, W] in [B * 1 * H * W] (un vettore 1D)
#     pred = pred.contiguous().view(-1)
#     target = target.contiguous().view(-1)
#
#     # 3. Calcolo
#     intersection = (pred * target).sum()
#     union = pred.sum() + target.sum()
#
#     dice = (2. * intersection + smooth) / (union + smooth)
#     return dice.item()


def compute_weight_map(labels, w0=10, sigma=5):
    """
    Implementazione fedele dell'Eq. 2 del paper U-Net:
    w(x) = w_c(x) + w_0 * exp(-(d_1(x) + d_2(x))^2 / (2*sigma^2))

    Dove:
    - w_c(x): peso per bilanciare le frequenze delle classi
    - d_1(x): distanza dal bordo della cella più vicina
    - d_2(x): distanza dal bordo della seconda cella più vicina
    - w_0=10, sigma≈5 pixels (valori del paper)

    Args:
        labels: tensor [B, 1, H, W] con 0=background, 1=membrane
        w0: peso per enfatizzare i bordi di separazione
        sigma: larghezza gaussiana

    Returns:
        weight_map: tensor [B, 1, H, W] con pesi per ogni pixel
    """
    from scipy.ndimage import distance_transform_edt
    from scipy.ndimage import label as scipy_label
    import numpy as np

    batch_size = labels.shape[0]
    weight_maps = []

    for b in range(batch_size):
        # Converti a numpy per scipy
        label_img = labels[b, 0].cpu().numpy()  # [H, W]

        # 1. w_c(x): Class frequency balancing
        # Calcola frequenze delle classi
        n_pixels = label_img.size
        n_membrane = (label_img == 1).sum()
        n_background = (label_img == 0).sum()

        # Pesi inversamente proporzionali alla frequenza
        w_background = n_pixels / (2 * n_background) if n_background > 0 else 1.0
        w_membrane = n_pixels / (2 * n_membrane) if n_membrane > 0 else 1.0

        w_c = np.where(label_img == 1, w_membrane, w_background)

        # 2. Identifica celle individuali (regioni connesse di background/cellule)
        # Nel dataset EM: 0=membrane (nero), 255=cell/background (bianco)
        # Invertiamo: cerchiamo regioni di celle (non membrane)
        cells = (label_img == 0).astype(np.uint8)  # celle = background

        # Etichetta componenti connesse
        labeled_cells, num_cells = scipy_label(cells)

        if num_cells < 2:
            # Se meno di 2 celle, nessun bordo di separazione da enfatizzare
            weight_map = w_c
        else:
            # 3. Calcola d_1 e d_2: distanze dalle celle più vicine
            # Per ogni pixel, trova le 2 celle più vicine

            # Array per memorizzare le distanze
            distances = np.zeros((num_cells, *label_img.shape))

            # Calcola distanza da ogni cella
            for cell_id in range(1, num_cells + 1):
                cell_mask = (labeled_cells == cell_id)
                if cell_mask.sum() > 0:
                    # Distance transform: distanza dal bordo della cella
                    # Per i pixel dentro la cella: 0
                    # Per i pixel fuori: distanza in pixel
                    distances[cell_id - 1] = distance_transform_edt(~cell_mask)

            # Ordina le distanze per ogni pixel
            distances_sorted = np.sort(distances, axis=0)

            # d_1: distanza dalla cella più vicina
            d1 = distances_sorted[0]

            # d_2: distanza dalla seconda cella più vicina
            d2 = distances_sorted[1] if num_cells > 1 else distances_sorted[0]

            # 4. Calcola il termine di enfatizzazione dei bordi
            # w_0 * exp(-(d_1 + d_2)^2 / (2*sigma^2))
            border_weight = w0 * np.exp(-((d1 + d2) ** 2) / (2 * sigma ** 2))

            # 5. Combina: w(x) = w_c(x) + border_weight
            weight_map = w_c + border_weight

        # Converti a tensor
        weight_maps.append(torch.from_numpy(weight_map).float())

    # Stack batch
    weight_maps = torch.stack(weight_maps).unsqueeze(1)  # [B, 1, H, W]
    return weight_maps.to(labels.device)


import numpy as np
import matplotlib.pyplot as plt
import torch
import tifffile
from scipy.ndimage import distance_transform_edt, label as scipy_label


def compute_weight_map_visualize(labels, w0=10, sigma=5):
    """
    Implementazione completa della weight map U-Net con output intermedi per debug
    """
    from scipy.ndimage import distance_transform_edt
    from scipy.ndimage import label as scipy_label

    # Converti a numpy
    label_img = labels.squeeze().cpu().numpy() if torch.is_tensor(labels) else labels

    # 1. Class frequency balancing (w_c)
    n_pixels = label_img.size
    n_membrane = (label_img == 1).sum()
    n_background = (label_img == 0).sum()

    w_background = n_pixels / (2 * n_background) if n_background > 0 else 1.0
    w_membrane = n_pixels / (2 * n_membrane) if n_membrane > 0 else 1.0

    w_c = np.where(label_img == 1, w_membrane, w_background)

    print(f"Class balancing weights:")
    print(f"  Background: {w_background:.3f}")
    print(f"  Membrane: {w_membrane:.3f}")

    # 2. Identifica celle individuali
    cells = (label_img == 0).astype(np.uint8)
    labeled_cells, num_cells = scipy_label(cells)

    print(f"Detected {num_cells} cells")

    if num_cells < 2:
        print("Less than 2 cells, no border emphasis needed")
        return w_c, w_c, None, None, labeled_cells

    # 3. Calcola distanze
    distances = np.zeros((num_cells, *label_img.shape))

    for cell_id in range(1, num_cells + 1):
        cell_mask = (labeled_cells == cell_id)
        if cell_mask.sum() > 0:
            distances[cell_id - 1] = distance_transform_edt(~cell_mask)

    distances_sorted = np.sort(distances, axis=0)
    d1 = distances_sorted[0]
    d2 = distances_sorted[1]

    # 4. Border weight
    border_weight = w0 * np.exp(-((d1 + d2) ** 2) / (2 * sigma ** 2))

    # 5. Final weight map
    weight_map = w_c + border_weight

    print(f"Weight map statistics:")
    print(f"  Min: {weight_map.min():.3f}")
    print(f"  Max: {weight_map.max():.3f}")
    print(f"  Mean: {weight_map.mean():.3f}")
    print(f"  Border weight max: {border_weight.max():.3f}")

    return weight_map, w_c, border_weight, (d1, d2), labeled_cells


def visualize_weight_map(image, labels, w0=10, sigma=5, save_path=None):
    """
    Visualizza tutti i componenti della weight map
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Calcola weight map con dettagli
    weight_map, w_c, border_weight, distances, labeled_cells = \
        compute_weight_map_visualize(labels, w0, sigma)

    # 1. Immagine originale
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # 2. Labels
    axes[0, 1].imshow(labels, cmap='gray')
    axes[0, 1].set_title('Ground Truth Labels\n(0=background/cell, 1=membrane)')
    axes[0, 1].axis('off')

    # 3. Celle identificate
    axes[0, 2].imshow(labeled_cells, cmap='tab20')
    axes[0, 2].set_title(f'Detected Cells (n={labeled_cells.max()})')
    axes[0, 2].axis('off')

    # 4. Class balancing weight (w_c)
    im3 = axes[0, 3].imshow(w_c, cmap='hot')
    axes[0, 3].set_title('Class Balancing Weight (w_c)')
    axes[0, 3].axis('off')
    plt.colorbar(im3, ax=axes[0, 3], fraction=0.046)

    if distances is not None:
        d1, d2 = distances

        # 5. Distance to nearest cell (d1)
        im4 = axes[1, 0].imshow(d1, cmap='viridis')
        axes[1, 0].set_title('Distance to Nearest Cell (d₁)')
        axes[1, 0].axis('off')
        plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)

        # 6. Distance to 2nd nearest cell (d2)
        im5 = axes[1, 1].imshow(d2, cmap='viridis')
        axes[1, 1].set_title('Distance to 2nd Nearest Cell (d₂)')
        axes[1, 1].axis('off')
        plt.colorbar(im5, ax=axes[1, 1], fraction=0.046)

        # 7. Border emphasis weight
        im6 = axes[1, 2].imshow(border_weight, cmap='hot')
        axes[1, 2].set_title(f'Border Weight\nw₀·exp(-(d₁+d₂)²/2σ²)\nw₀={w0}, σ={sigma}')
        axes[1, 2].axis('off')
        plt.colorbar(im6, ax=axes[1, 2], fraction=0.046)
    else:
        for ax in [axes[1, 0], axes[1, 1], axes[1, 2]]:
            ax.axis('off')

    # 8. Final weight map
    im7 = axes[1, 3].imshow(weight_map, cmap='hot')
    axes[1, 3].set_title('Final Weight Map\nw(x) = w_c(x) + border_weight')
    axes[1, 3].axis('off')
    plt.colorbar(im7, ax=axes[1, 3], fraction=0.046)

    plt.suptitle('U-Net Weight Map Computation (Equation 2)', fontsize=16, y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.show()


def main():
    """
    Test della weight map su un'immagine del dataset EM
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--volume', default='dataset/train-volume.tif',
                        help='Path to training volume')
    parser.add_argument('--labels', default='dataset/train-labels.tif',
                        help='Path to training labels')
    parser.add_argument('--idx', type=int, default=0,
                        help='Image index to visualize')
    parser.add_argument('--w0', type=float, default=10,
                        help='w0 parameter (default: 10)')
    parser.add_argument('--sigma', type=float, default=5,
                        help='sigma parameter (default: 5)')
    parser.add_argument('--save', type=str, default=None,
                        help='Path to save visualization')

    args = parser.parse_args()

    # Carica dati
    print(f"Loading data from {args.volume}...")
    volume = tifffile.imread(args.volume)
    labels = tifffile.imread(args.labels)

    print(f"Volume shape: {volume.shape}")
    print(f"Labels shape: {labels.shape}")

    # Seleziona immagine
    image = volume[args.idx]
    label = labels[args.idx]

    # Converti label: 0=membrane, 255=cell -> 0=cell, 1=membrane
    label_binary = (label == 0).astype(np.float32)

    print(f"\nProcessing image {args.idx}...")
    print(f"Image range: [{image.min()}, {image.max()}]")
    print(f"Label values: {np.unique(label)}")
    print(f"Membrane pixels: {(label_binary == 1).sum()} / {label_binary.size}")

    # Visualizza
    visualize_weight_map(
        image,
        label_binary,
        w0=args.w0,
        sigma=args.sigma,
        save_path=args.save
    )


if __name__ == '__main__':
    main()