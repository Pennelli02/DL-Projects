

import os
from types import SimpleNamespace
import argparse
import logging
from ipdb import launch_ipdb_on_exception
import yaml
from rich.logging import RichHandler
import numpy as np
import torch

from ViT import ViT, PretrainedViT
from src.utils import EarlyStopping


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


def save_checkpoint(model, optimizer, epoch, loss, opts):
    os.makedirs(opts.checkpoint_dir, exist_ok=True)
    fname = os.path.join(opts.checkpoint_dir, f'e_{epoch:05d}.chp')
    info = dict(
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        epoch=epoch,
        loss=loss
    )
    torch.save(info, fname)
    LOG.info(f'Saved checkpoint {fname}')


def test_metrics(model, test, opts):
    model.eval()
    correct = []

    with torch.no_grad():
        for Xcpu, Y in test:
            X = Xcpu.to(opts.device)
            Y = Y.to(opts.device)
            out = model(X)

            predictions = np.argmax(N(out), axis=1)
            labels = N(Y)
            c = list(predictions == labels)
            correct.extend(c)

    return np.mean(correct)


def train_loop(model, train, valid, opts):
    import tensorflow as tf

    os.makedirs(f'tensorboard/{opts.model_structure}', exist_ok=True)

    train_writer = tf.summary.create_file_writer(f'tensorboard/{opts.model_structure}/train')
    test_writer = tf.summary.create_file_writer(f'tensorboard/{opts.model_structure}/validation')

    # OPTIMIZER corretto secondo articolo
    if opts.model_structure == 'backbone':
        # Fine-tuning: usa SGD con momentum (come articolo)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=opts.lr,
            momentum=opts.momentum,
            weight_decay=opts.weight_decay
        )
        LOG.info(f"Using SGD optimizer (lr={opts.lr}, momentum={opts.momentum})")
    else:
        # Training from scratch: usa Adam (come articolo per pre-training)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=opts.lr,
            betas=(0.9, 0.999),
            weight_decay=opts.weight_decay
        )
        LOG.info(f"Using Adam optimizer (lr={opts.lr})")

    # SCHEDULER corretto secondo articolo
    if opts.scheduler == 'cosine':
        # Articolo Tabella 4: cosine decay per fine-tuning
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=opts.num_epochs
        )
        LOG.info("Using CosineAnnealingLR scheduler")
    elif opts.scheduler == 'linear':
        # Articolo Tabella 3: linear decay per pre-training
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: 1 - epoch / opts.num_epochs
        )
        LOG.info("Using Linear decay scheduler")
    else:
        scheduler = None
        LOG.info("No scheduler")

    loss_function = torch.nn.CrossEntropyLoss()

    # creiamo una classe che implementa l'early stopping
    early_stopping = EarlyStopping(patience=3, min_delta=0.001, save_path="ViT_best_model.chp")
    step = 0
    for epoch in range(1, opts.num_epochs + 1):
        model.train()
        mses = []
        correct = []
        batch_i = 0

        for Xcpu, Y in train:
            batch_i += 1
            X = Xcpu.to(opts.device)
            Y = Y.to(opts.device)

            optimizer.zero_grad()
            out = model(X)
            loss = loss_function(out, Y)
            loss.backward()

            # Gradient clipping (articolo: global norm 1)
            if hasattr(opts, 'grad_clip'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), opts.grad_clip)

            optimizer.step()

            mses.append(N(loss))
            c = np.mean(np.argmax(N(out), axis=1) == N(Y))
            correct.append(c)

            if batch_i % opts.log_every == 0:
                train_l = np.mean(mses[-opts.batch_window:])
                train_a = np.mean(correct[-opts.batch_window:])
                test_a = test_metrics(model, valid, opts)

                msg = f'{epoch:03d}.{batch_i:03d}: '
                msg += f'train: loss={train_l:1.6f}, acc={train_a:1.3f} '
                msg += f'test: acc={test_a:1.3f}'
                LOG.info(msg)

                with train_writer.as_default():
                    tf.summary.scalar('loss', train_l, step=step)
                    tf.summary.scalar('accuracy', train_a, step=step)
                with test_writer.as_default():
                    tf.summary.scalar('accuracy', test_a, step=step)
                step += 1

            # --- Early Stopping ---
        early_stopping(test_a, model)
        if early_stopping.early_stop:
           LOG.info(f"Training interrotto all'epoca {epoch} per Early Stopping")
           break

        if epoch % opts.save_every == 0:
            save_checkpoint(model, optimizer, epoch, loss, opts)

        # Step dello scheduler DOPO ogni epoca
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            LOG.info(f'Epoch {epoch} completed. Learning rate: {current_lr:.6f}')


def main(opts):
    from visualizer import visualize
    from dataset import MakeDataloader

    input_data = torch.randn(opts.batch_size, 3, opts.img_size, opts.img_size)
    num_classes = opts.num_classes

    if opts.model_structure == 'manual':
        model = ViT(num_classes=num_classes, img_size=opts.img_size)
    elif opts.model_structure == 'backbone':
        model = PretrainedViT(num_classes=num_classes)
    else:
        raise ValueError(f'Unknown model type {opts.model_structure}')

    visualize(model, "ViT", input_data)
    model = model.to(opts.device)

    if opts.model_structure == 'manual':
        decals = MakeDataloader(opts, opts.dataset_path, manual=True)
    elif opts.model_structure == 'backbone':
        decals = MakeDataloader(
            opts,
            opts.dataset_path,
            manual=False,
            transform=model.get_transforms()
        )

    train = decals.train_dataloader
    valid = decals.test_dataloader

    train_loop(model, train, valid, opts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help='YAML Configuration file')
    opts = yaml.load(open(parser.parse_args().config), Loader=yaml.Loader)
    opts = SimpleNamespace(**opts)
    opts.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with launch_ipdb_on_exception():
        main(opts)