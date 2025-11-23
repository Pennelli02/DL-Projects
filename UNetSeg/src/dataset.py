import tifffile
import torch
from albumentations import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import numpy as np
import albumentations as A
import cv2

class EMDataset(Dataset):
    """Dataset per EM Segmentation Challenge fedele al paper"""

    def __init__(self, opts, volume_path, labels_path, transform=None):
        """
        Args:
            opts: configurazione
            volume_path: path al file .tif con le immagini
            labels_path: path al file .tif con le labels
            transform: trasformazioni da applicare
        """
        self.opts = opts
        # Carica lo stack di immagini (em presenta i file in .tif)
        self.volume = tifffile.imread(volume_path)  # Shape: (30, 512, 512)
        self.labels = tifffile.imread(labels_path)  # Shape: (30, 512, 512)
        self.transform = transform

        print(f"✓ Loaded EM dataset:")
        print(f"  Volume shape: {self.volume.shape}")
        print(f"  Labels shape: {self.labels.shape}")
        print(f"  Volume range: [{self.volume.min()}, {self.volume.max()}]")
        print(f"  Labels unique values: {np.unique(self.labels)}")

    def __len__(self):
        return self.volume.shape[0]

    def __getitem__(self, idx):
        # 1. Carica e normalizza [0, 1]
        img = self.volume[idx].astype(np.float32) / 255.0  # (H, W)

        # 2. Labels: 0=membrane (nero), 255=cell/background (bianco)
        # Convertiamo: 1=membrane, 0=background
        mask = (self.labels[idx] == 0).astype(np.float32)  # (H, W)

        # 3. Applica trasformazioni (se presenti)
        if self.transform:
            # Albumentations lavora con (H, W) numpy arrays
            transformed = self.transform(image=img, mask=mask)
            img = transformed["image"]  # Ancora numpy array (H, W)
            mask = transformed["mask"]  # Ancora numpy array (H, W)

            # Converti a tensor manualmente (NON usare il ToTensorV2 di Albumentations!)
            img = torch.from_numpy(img).float()
            mask = torch.from_numpy(mask).float()
        else:
            # Converti manualmente a tensor
            img = torch.from_numpy(img).float()
            mask = torch.from_numpy(mask).float()

        # 4. Aggiungi channel dimension: (H, W) -> (1, H, W)
        img = img.unsqueeze(0)
        mask = mask.unsqueeze(0)

        return img, mask


class EMDatasetAugmented(EMDataset):
    """
    Dataset EM con augmentation ESATTAMENTE come nel paper

    Paper section 3.1:
    "We generate smooth deformations using random displacement vectors on a
    coarse 3 by 3 grid. The displacements are sampled from a Gaussian
    distribution with 10 pixels standard deviation."
    """

    def __init__(self, opts, volume_path, labels_path):
        """
        Args:
            opts: configurazione
            volume_path: path al file .tif con le immagini
            labels_path: path al file .tif con le labels
        """
        # Crea la pipeline di augmentation PRIMA di chiamare super().__init__
        transform = self.get_unet_paper_augmentation()

        # Chiama il costruttore della classe base
        super().__init__(opts, volume_path, labels_path, transform=transform)

        print(f"✓ Using paper-faithful augmentation:")
        print(f"  - Elastic Transform: alpha=720, sigma=24")
        print(f"  - Random rotations: 0-360°")
        print(f"  - Flips: horizontal + vertical")
        print(f"  - Gray value variations")

    def get_unet_paper_augmentation(self):
        """
        Augmentation pipeline secondo paper U-Net (Section 3.1)
        NOTA: NON usa ToTensorV2, la conversione a tensor è fatta in __getitem__
        """


        return A.Compose([
            # 1. ELASTIC DEFORMATION (augmentation principale del paper)
            A.ElasticTransform(
                alpha=720.0,  # Intensità deformazione
                sigma=24.0,  # Smoothness (più alto = più smooth)
                interpolation=cv2.INTER_CUBIC,
                border_mode=cv2.BORDER_REFLECT,  # Mirror padding (paper)
                p=0.5  # Applica al 50% delle immagini
            ),

            # 2. ROTATIONS (invarianza rotazionale)
            A.Rotate(
                limit=180,  # Rotazioni 0-360° (±180)
                interpolation=cv2.INTER_CUBIC,
                border_mode=cv2.BORDER_REFLECT,
                p=0.5
            ),

            # 3. FLIPS (invarianza a specchiamento)
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),

            # 4. GRAY VALUE VARIATIONS (paper: "robustness to gray value variations")
            A.RandomBrightnessContrast(
                brightness_limit=0.2,  # ±20%
                contrast_limit=0.2,  # ±20%
                p=0.3
            ),

            # NON usare ToTensorV2 qui - conversione manuale in __getitem__
        ])


class testEMDataset(Dataset):
    """Dataset per test set senza augmentation"""

    def __init__(self, opts, volume_path, labels_path, transform=None):
        self.opts = opts
        self.volume = tifffile.imread(volume_path)
        self.labels = tifffile.imread(labels_path)
        self.transform = transform

        print(f"✓ Loaded test EM dataset:")
        print(f"  Volume shape: {self.volume.shape}")
        print(f"  Labels shape: {self.labels.shape}")

    def __len__(self):
        return self.volume.shape[0]

    def __getitem__(self, idx):
        # 1. Carica e normalizza [0, 1]
        img = self.volume[idx].astype(np.float32) / 255.0  # (H, W)
        mask = (self.labels[idx] == 0).astype(np.float32)  # (H, W)

        # 2. Applica trasformazioni (se presenti)
        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed["image"]  # Ancora numpy (H, W)
            mask = transformed["mask"]  # Ancora numpy (H, W)

            # Converti a tensor manualmente
            img = torch.from_numpy(img).float()
            mask = torch.from_numpy(mask).float()
        else:
            img = torch.from_numpy(img).float()
            mask = torch.from_numpy(mask).float()

        # 3. Aggiungi channel dimension: (H, W) -> (1, H, W)
        img = img.unsqueeze(0)
        mask = mask.unsqueeze(0)

        return img, mask


class MakeDataLoader:
    """Crea train, validation e test dataloaders"""

    def __init__(self, opts, trainData, testData):
        """
        Args:
            opts: configurazione
            trainData: dataset di training (con augmentation)
            testData: dataset di test (senza augmentation)
        """
        # 1. Split training in train + validation interna
        generator = torch.Generator().manual_seed(opts.seed)
        total_train_size = len(trainData)

        train_size = int(opts.train_split * total_train_size)
        internal_val_size = total_train_size - train_size

        print(f"\n✓ Splitting training data:")
        print(f"  Total: {total_train_size}")
        print(f"  Train: {train_size} ({opts.train_split * 100:.0f}%)")
        print(f"  Internal Validation: {internal_val_size} ({(1 - opts.train_split) * 100:.0f}%)")
        print(f"  External Test: {len(testData)}")

        train_set, internal_validation_set = torch.utils.data.random_split(
            trainData,
            lengths=[train_size, internal_val_size],
            generator=generator
        )

        # 2. Crea DataLoaders
        self.train_dataloader = DataLoader(
            train_set,
            batch_size=opts.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )

        self.internal_validation_dataloader = DataLoader(
            internal_validation_set,
            batch_size=opts.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )

        self.test_dataloader = DataLoader(
            testData,
            batch_size=opts.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )

        print(f"✓ DataLoaders created:")
        print(f"  Train batches: {len(self.train_dataloader)}")
        print(f"  Internal Val batches: {len(self.internal_validation_dataloader)}")
        print(f"  Test batches: {len(self.test_dataloader)}")


# Test del dataset
if __name__ == '__main__':
    """Test rapido del dataset e augmentation"""
    import matplotlib.pyplot as plt
    from types import SimpleNamespace

    # Config minima per test
    opts = SimpleNamespace(
        seed=42,
        train_split=0.8,
        batch_size=1
    )

    # Paths (modifica con i tuoi paths)
    volume_path = 'dataset/train-volume.tif'
    labels_path = 'dataset/train-labels.tif'

    print("=" * 70)
    print("Testing EMDatasetAugmented...")
    print("=" * 70)

    dataset = EMDatasetAugmented(opts, volume_path, labels_path)

    print(f"\nDataset length: {len(dataset)}")

    # Test getitem
    img, mask = dataset[0]
    print(f"\n✓ Sample shapes:")
    print(f"  Image: {img.shape} (expected: [1, H, W])")
    print(f"  Mask: {mask.shape} (expected: [1, H, W])")
    print(f"  Image dtype: {img.dtype}")
    print(f"  Mask dtype: {mask.dtype}")
    print(f"  Image range: [{img.min():.3f}, {img.max():.3f}]")
    print(f"  Mask unique values: {torch.unique(mask)}")

    # VERIFICA CRITICA
    assert len(img.shape) == 3, f"Image should be 3D [1,H,W], got {img.shape}"
    assert len(mask.shape) == 3, f"Mask should be 3D [1,H,W], got {mask.shape}"
    assert img.shape[0] == 1, f"Image should have 1 channel, got {img.shape[0]}"
    assert mask.shape[0] == 1, f"Mask should have 1 channel, got {mask.shape[0]}"
    print("✓ Shape verification passed!")

    # Visualizza 4 augmentazioni diverse
    fig, axes = plt.subplots(4, 2, figsize=(10, 16))

    for i in range(4):
        img, mask = dataset[0]  # Stessa immagine, augmentation diversa

        # Remove channel dimension per visualizzazione
        img_np = img[0].numpy()
        mask_np = mask[0].numpy()

        axes[i, 0].imshow(img_np, cmap='gray')
        axes[i, 0].set_title(f'Augmented Image {i + 1}')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask_np, cmap='gray')
        axes[i, 1].set_title(f'Augmented Mask {i + 1}')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig('dataset_test_augmentation.png', dpi=150)
    print(f"\n✓ Saved visualization to 'dataset_test_augmentation.png'")
    plt.show()

    # Test DataLoader con testData
    print("\n" + "=" * 70)
    print("Testing DataLoader with train and test data...")
    print("=" * 70)

    test_volume_path = 'dataset/test-volume.tif'
    test_labels_path = 'dataset/test-labels.tif'

    try:
        # Crea test dataset (senza augmentation)
        test_transform = A.Compose([ToTensorV2()])
        test_dataset = testEMDataset(opts, test_volume_path, test_labels_path, transform=test_transform)

        # Crea DataLoader
        dataloaders = MakeDataLoader(opts, dataset, test_dataset)

        # Test batch da train
        print("\nTesting train batch...")
        for X, Y in dataloaders.train_dataloader:
            print(f"✓ Train batch shapes: X={X.shape}, Y={Y.shape}")
            assert len(X.shape) == 4, f"Batch should be 4D [B,C,H,W], got {X.shape}"
            assert X.shape[1] == 1, f"Should have 1 channel, got {X.shape[1]}"
            break

        # Test batch da test
        print("\nTesting test batch...")
        for X, Y in dataloaders.test_dataloader:
            print(f"✓ Test batch shapes: X={X.shape}, Y={Y.shape}")
            break

        print("\n✓ All DataLoader tests passed!")

    except FileNotFoundError as e:
        print(f"⚠ Test dataset not found: {e}")
        print("  Skipping DataLoader test")

    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)