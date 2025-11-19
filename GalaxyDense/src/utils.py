# utils.py
import torch
import numpy as np

# Presente nell'articolo serve per indicare i pesi di una classe
from sklearn.utils.class_weight import compute_class_weight


def calculate_class_weights(dataloader, num_classes=10, device='cuda'):
    """
    Calcola i class weights da un DataLoader.

    Args:
        dataloader: DataLoader del training set
        num_classes: numero di classi (10 per Galaxy10)
        device: dispositivo per il tensor

    Returns:
        torch.Tensor con i pesi per ogni classe
    """
    print("\n" + "=" * 80)
    print("CALCOLO CLASS WEIGHTS")
    print("=" * 80)

    # Estrai tutte le labels dal dataloader
    all_labels = []

    # ---------------------------------------------------------------------
    # 🚨 CORREZIONE: 'labels' sono ora Label Encoded (indici interi)
    # Rimuoviamo torch.argmax(..., dim=1) che causava l'IndexError.
    # ---------------------------------------------------------------------
    for _, labels in dataloader:
        # Spostiamo il tensore su CPU e convertiamo direttamente in NumPy
        # senza usare argmax, poiché 'labels' contiene già gli indici.
        labels_indices = labels.cpu().numpy()
        all_labels.extend(labels_indices)

    all_labels = np.array(all_labels)

    # Calcola i weights usando sklearn
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(all_labels),
        y=all_labels
    )

    # Stampa statistiche
    print("\nDistribuzione classi e pesi:")
    print(f"{'Classe':<10} {'N° Campioni':<15} {'Peso':<10}")
    print("-" * 40)
    for i in range(num_classes):
        # Aggiustiamo per il caso in cui compute_class_weight restituisca meno di 10 pesi
        # (se una classe è assente nel dataset - anche se qui non dovrebbe)
        if i < len(class_weights):
            n_samples = np.sum(all_labels == i)
            weight = class_weights[i]
            print(f"{i:<10} {n_samples:<15} {weight:<10.3f}")
        else:
            print(f"{i:<10} {'0':<15} {'0.000':<10}")  # Classe mancante

    print("=" * 80 + "\n")

    # Converti in tensor PyTorch
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    return class_weights_tensor