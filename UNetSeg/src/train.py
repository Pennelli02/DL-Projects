import os
from types import SimpleNamespace
import argparse
import logging
import torch.nn.functional as F
from ipdb import launch_ipdb_on_exception
import yaml
from rich.logging import RichHandler
import numpy as np
import torch

from src.unet import UNet
from src.utils import compute_weight_map


def get_logger():
    FORMAT = "%(message)s"
    logging.basicConfig(
        level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
    )
    log = logging.getLogger("rich")
    return log


LOG = get_logger()

def N(x):
    return x.detach().cpu().numpy()


def center_crop_tensor(data, output_size):
    """Ritaglia il tensore centralmente per adattarsi a output_size (H, W)."""
    _, _, H, W = data.shape
    H_out, W_out = output_size  # output_size sarà un tuple (H_out, W_out)

    # Calcola il margine da tagliare (sempre pari)
    diff_y = H - H_out
    diff_x = W - W_out

    # Calcola gli indici di inizio e fine
    crop_start_y = diff_y // 2
    crop_end_y = H - (diff_y - crop_start_y)

    crop_start_x = diff_x // 2
    crop_end_x = W - (diff_x - crop_start_x)

    # Esegue il ritaglio e restituisce il tensore ritagliato
    return data[:, :, crop_start_y:crop_end_y, crop_start_x:crop_end_x]

def save_checkpoint(model, optimizer, epoch, loss, opts):
    fname = os.path.join(opts.checkpoint_dir,f'e_{epoch:05d}.chp')
    info = dict(model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                epoch=epoch, loss=loss)
    torch.save(info, fname)
    LOG.info(f'Saved checkpoint {fname}')


def dice_from_preds_and_target(pred_logits, target, smooth=1e-6):
    """
    pred_logits: [B, 2, H, W]  (logits)
    target: [B, 1, H, W] (0/1)
    returns dice per batch (float)
    """
    # Pred class via argmax
    preds = torch.argmax(pred_logits, dim=1, keepdim=True).float()  # [B,1,H,W], values 0/1
    target = target.float()
    preds_flat = preds.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    intersection = (preds_flat * target_flat).sum()
    union = preds_flat.sum() + target_flat.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.item()


def test_metrics(model, test_loader, opts):
    model.eval()
    total_dice = 0.0
    total_pixel_acc = 0.0
    num_batches = 0

    with torch.no_grad():
        for X, Y in test_loader:
            X = X.to(opts.device)
            Y = Y.to(opts.device)

            out = model(X)  # [B,2,H_out,W_out]
            # crop Y to match out
            _, _, H_out, W_out = out.shape
            Y_cropped = center_crop_tensor(Y, (H_out, W_out))

            # Dice (class 1 = membrane)
            dice = dice_from_preds_and_target(out, Y_cropped)
            total_dice += dice

            # Pixel accuracy
            preds = torch.argmax(out, dim=1, keepdim=True)  # [B,1,H,W]
            pixel_acc = (preds == Y_cropped).float().mean().item()
            total_pixel_acc += pixel_acc

            num_batches += 1

    avg_dice = total_dice / max(1, num_batches)
    avg_pixel_acc = total_pixel_acc / max(1, num_batches)
    return avg_dice, avg_pixel_acc


def train_loop(model, train_loader, valid_loader, opts):
    import tensorflow as tf

    # TensorBoard writers
    train_writer = tf.summary.create_file_writer(f'tensorboard/unet/train')
    test_writer = tf.summary.create_file_writer(f'tensorboard/unet/validation')

    # Optimizer (paper usa SGD con momentum 0.99)
    if opts.use_sgd:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=opts.lr,
            momentum=opts.momentum,
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr)

    # Loss function
    # Paper: pixel-wise softmax + cross entropy con weight map
    # Per segmentazione binaria: BCE with logits (senza reduction per applicare weight map)
    #loss_function = torch.nn.BCEWithLogitsLoss(reduction='none')

    step = 0
    best_dice = 0.0

    for epoch in range(1, opts.num_epochs + 1):
        model.train()
        epoch_losses = []
        epoch_dice = []

        for batch_i, (X, Y) in enumerate(train_loader, 1):
            X = X.to(opts.device)  # [B, 1, H, W]
            Y = Y.to(opts.device)  # [B, 1, H, W]

            optimizer.zero_grad()

            out = model(X)  # [B, 2, H, W]

            # Estrai il canale membrane (channel 0)
            pred_membrane = out[:, 0:1, :, :]  # [B, 1, H, W]
            # Ottieni le dimensioni spaziali dell'output
            _, _, H_out, W_out = pred_membrane.shape
            Y_cropped = center_crop_tensor(Y, (H_out, W_out))  # Y_cropped: [B, 1, H_out, W_out]

            # DEBUG
            # # 1) Mostra range input e shape
            # print("DEBUG: X range:", float(X.min()), float(X.max()))
            # print("DEBUG: out.shape:", out.shape)
            # print("DEBUG: Y.shape:", Y.shape)
            # print("DEBUG: Y_cropped.shape:", Y_cropped.shape)
            #
            # # 2) Controlla valori unici del target
            # print("DEBUG: unique Y:", torch.unique(Y))
            # print("DEBUG: unique Y_cropped:", torch.unique(Y_cropped))
            #
            # # 3) Calcola softmax e stampa min/max probabilità del canale membrana
            # probs = torch.softmax(out, dim=1)  # [B,2,H,W]
            # print("DEBUG: probs channel 0 (background) min/max:", float(probs[0, 0].min()),
            #           float(probs[0, 0].max()))
            # print("DEBUG: probs channel 1 (membrane) min/max:", float(probs[0, 1].min()), float(probs[0, 1].max()))
            #
            # # 4) Controlla argmax e quanti pixel vengono predetti come membrana
            # preds = torch.argmax(out, dim=1)
            # print("DEBUG: membrane pixels predicted:", (preds == 1).sum().item(), "/", preds.numel())
            #
            # # 5) Calcola Dice manualmente
            # dice_manual = (2 * ((preds == 1) & (Y_cropped.squeeze(1) == 1)).sum()) / \
            #                   ((preds == 1).sum() + (Y_cropped.squeeze(1) == 1).sum() + 1e-6)
            # print("DEBUG: dice manual:", float(dice_manual))

            # 1) prepara il target per CrossEntropy
            Y_target_ce = Y_cropped.squeeze(1).long()  # [B, H_out, W_out], values in {0,1}

            # 2) loss per-pixel (senza reduction)
            loss_per_pixel = F.cross_entropy(out, Y_target_ce, reduction='none')  # [B, H_out, W_out]

            # Calcola weight map secondo paper (Eq. 2)
            if opts.use_weight_map:
                # compute_weight_map returns [B,1,H,W]
                weight_map = compute_weight_map(Y_cropped, w0=10, sigma=5).squeeze(1)  # [B,H_out,W_out]
                loss = (loss_per_pixel * weight_map).mean()
            else:
                loss = loss_per_pixel.mean()

            loss.backward()
            optimizer.step()

            # Metrics
            # Metrics
            epoch_losses.append(loss.item())

            # usa out (tutti i canali) per il dice con argmax
            dice = dice_from_preds_and_target(out, Y_cropped)  # out: [B,2,H_out,W_out], Y_cropped: [B,1,H_out,W_out]
            epoch_dice.append(dice)

            # Logging periodico
            if batch_i % opts.log_every == 0:
                train_loss = np.mean(epoch_losses[-opts.batch_window:])
                train_dice = np.mean(epoch_dice[-opts.batch_window:])

                # Validation metrics
                val_dice, val_pixel_acc = test_metrics(model, valid_loader, opts)

                msg = f'{epoch:03d}.{batch_i:03d}: '
                msg += f'train: loss={train_loss:.6f}, dice={train_dice:.3f} '
                msg += f'val: dice={val_dice:.3f}, pixel_acc={val_pixel_acc:.3f}'
                LOG.info(msg)

                # TensorBoard logging
                with train_writer.as_default():
                    tf.summary.scalar('loss', train_loss, step=step)
                    tf.summary.scalar('dice', train_dice, step=step)
                with test_writer.as_default():
                    tf.summary.scalar('dice', val_dice, step=step)
                    tf.summary.scalar('pixel_accuracy', val_pixel_acc, step=step)

                step += 1

                # Salva best model
                if val_dice > best_dice:
                    best_dice = val_dice
                    save_checkpoint(model, optimizer, epoch, loss.item(), opts)
                    LOG.info(f'✓ New best Dice: {best_dice:.4f}')

        # Checkpoint periodico
        if epoch % opts.save_every == 0:
            save_checkpoint(model, optimizer, epoch, loss.item(), opts)



def main(opts):
    from dataset import EMDataset, EMDatasetAugmented, MakeDataLoader, testEMDataset
    from visualizer import visualize
    input_data = torch.randn(opts.batch_size, 1, 512 ,512 )
    # Crea modello U-Net
    model = UNet()
    LOG.info(f"Model created: {sum(p.numel() for p in model.parameters())} parameters")
    visualize(model, 'DenseNet', input_data)
    model = model.to(opts.device)

    # Carica dataset
    if opts.augmentation:
        data = EMDatasetAugmented(
            opts,
            opts.volume_path,
            opts.labels_path
        )
    else:
        data = EMDataset(
            opts,
            opts.volume_path,
            opts.labels_path
        )

    test = testEMDataset(
            opts,
            opts.test_volume_path,
            opts.test_labels_path
    )
    LOG.info(f"Dataset loaded: {len(data)} images")

    # Data loaders
    dataloaders = MakeDataLoader(opts, data, test)
    train_loader = dataloaders.train_dataloader
    valid_loader = dataloaders.test_dataloader

    LOG.info(f"Train batches: {len(train_loader)}, Valid batches: {len(valid_loader)}")

    # Training
    train_loop(model, train_loader, valid_loader, opts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help='YAML Configuration file')
    opts = yaml.load(open(parser.parse_args().config), Loader=yaml.Loader)
    opts = SimpleNamespace(**opts)
    opts.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Crea checkpoint directory
    os.makedirs(opts.checkpoint_dir, exist_ok=True)

    with launch_ipdb_on_exception():
        main(opts)