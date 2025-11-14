import os
from types import SimpleNamespace
import argparse
import logging
from Parameters import config
from ipdb import launch_ipdb_on_exception

from rich.logging import RichHandler
import numpy as np
import torch

from ModelCNN import SimpleCNN


def get_logger():
    """
    Configura e restituisce un logger con formattazione colorata tramite Rich.
    Usato per stampare messaggi informativi durante il training.
    """
    FORMAT = "%(message)s"
    logging.basicConfig(
        level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
    )
    log = logging.getLogger("rich")
    return log


LOG = get_logger()


def N(x):
    """
    Converte un tensore PyTorch in array NumPy.
    Stacca il tensore dal grafo computazionale e lo sposta su CPU.
    Utile per calcolare metriche o visualizzazioni.
    """
    return x.detach().cpu().numpy()


def save_checkpoint(model, optimizer, epoch, loss, opts):
    """
    Salva uno snapshot del training (checkpoint).
    Include: pesi del modello, stato dell'optimizer, epoca corrente e loss.
    Permette di riprendere il training da dove si era interrotto.
    """
    fname = os.path.join(opts.checkpoint_dir, f'e_{epoch:05d}.chp')
    info = dict(model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                epoch=epoch, loss=loss)
    torch.save(info, fname)
    LOG.info(f'Saved checkpoint {fname}')


def test_metrics(model, test, device):
    """
    Calcola l'accuratezza del modello sul test set.

    Passaggi:
    1. Mette il modello in modalità eval (disabilita dropout/batch norm)
    2. Disabilita il calcolo dei gradienti per risparmiare memoria
    3. Per ogni batch: fa predizioni e confronta con le label vere
    4. Restituisce la percentuale di predizioni corrette
    5. Rimette il modello in modalità training
    """
    model.eval()
    correct = []
    with torch.no_grad():
        for Xcpu, Y in test:
            X = Xcpu.to(device)
            Y = Y.to(device)
            out = model(X)
            # Confronta le classi predette (argmax degli output) con le classi vere (argmax dei one-hot)
            c = list(np.argmax(N(out), axis=1) == np.argmax(N(Y), axis=1))
            correct.extend(c)
    model.train()
    return np.mean(correct)


def train_loop(model, train, test, opts):
    """
    Loop principale di training del modello.

    Per ogni epoca:
    1. Itera su tutti i batch del training set
    2. Per ogni batch:
       - Forward pass: calcola le predizioni
       - Calcola la loss (cross-entropy)
       - Backward pass: calcola i gradienti
       - Aggiorna i pesi con l'optimizer
    3. Ogni N batch logga le metriche (loss e accuracy) su TensorBoard
    4. Salva checkpoint periodicamente
    5. Riduce il learning rate con lo scheduler
    """
    import tensorflow as tf
    train_writer = tf.summary.create_file_writer(f'tensorboard/{opts.model}/train')
    test_writer = tf.summary.create_file_writer(f'tensorboard/{opts.model}/test')

    # Optimizer: AdamW con weight decay per la regolarizzazione
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=opts.learning_rate,
                                  weight_decay=opts.weight_decay)
    # Scheduler: riduce il learning rate del 10% ogni epoca (gamma=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # Loss function: cross-entropy per classificazione multi-classe
    loss_function = torch.nn.CrossEntropyLoss()
    step = 0

    for epoch in range(1, opts.num_epochs + 1):
        model.train()
        mses = []  # Salva le loss per calcolare la media
        correct = []  # Salva le accuratezze per calcolare la media
        batch_i = 0

        for Xcpu, Y in train:
            batch_i += 1
            X = Xcpu.to(opts.device)  # Sposta i dati sul device (CPU o GPU)
            Y = Y.to(opts.device)

            optimizer.zero_grad()  # Azzera i gradienti dell'iterazione precedente
            out = model(X)  # Forward pass: calcola le predizioni

            # Converti Y da one-hot a indici di classe (es. [0,0,1,0] -> 2)
            Y_indices = torch.argmax(Y, dim=1)
            loss = loss_function(out, Y_indices)  # Calcola la loss

            loss.backward()  # Backward pass: calcola i gradienti
            optimizer.step()  # Aggiorna i pesi

            # Salva metriche
            mses.append(N(loss))
            c = np.mean(np.argmax(N(out), axis=1) == N(Y_indices))
            correct.append(c)

            # Ogni N batch, logga le metriche
            if batch_i % opts.log_every == 0:
                train_l = np.mean(mses[-opts.batch_window:])  # Media delle ultime N loss
                train_a = np.mean(correct[-opts.batch_window:])  # Media delle ultime N accuracy
                test_a = test_metrics(model, test, opts.device)  # Calcola accuracy sul test set

                msg = f'{epoch:03d}.{batch_i:03d}: '
                msg += f'train: loss={train_l:1.6f}, acc={train_a:1.3f} '
                msg += f'test: acc={test_a:1.3f}'
                LOG.info(msg)

                # Salva su TensorBoard per visualizzazione
                with train_writer.as_default():
                    tf.summary.scalar('loss', train_l, step=step)
                    tf.summary.scalar('accuracy', train_a, step=step)
                with test_writer.as_default():
                    tf.summary.scalar('accuracy', test_a, step=step)
                step += 1

        # Salva checkpoint periodicamente
        if epoch % opts.save_every == 0:
            save_checkpoint(model, optimizer, epoch, loss, opts)

        # Riduce il learning rate
        scheduler.step()
        print(f"Learning rate: {scheduler.get_last_lr()}")


def main(opts):
    """
    Funzione principale che orchestra tutto il workflow:

    1. Carica il modello CNN
    2. Visualizza l'architettura con torchviz
    3. Carica il dataset (con o senza augmentation)
    4. Crea i DataLoader per training e test
    5. Avvia il loop di training
    """
    from visualizer import visualize
    from GalaxyDataset import GalaxyDataset
    from AugmentedGalaxyDataset import MakeDataLoaders, AugmentedGalaxyDataset

    input_size = 224
    input_data = torch.randn(opts.batch_size, 3, input_size, input_size)
    num_classes = 10

    # Crea il modello CNN
    model = SimpleCNN(num_filters=opts.simple_num_filters,
                      mlp_size=opts.simple_mlp_size,
                      num_classes=num_classes)
    # Visualizza l'architettura e salva il grafo
    visualize(model, 'myCNN', input_data)
    # Sposta il modello sul device (CPU o GPU)
    model = model.to(opts.device)

    # Crea il dataset con lazy loading (carica le immagini on-demand per risparmiare RAM)
    if opts.augmentation:
        data = AugmentedGalaxyDataset(opts, 'Dataset/Galaxy10_DECals.h5')
    else:
        data = GalaxyDataset(opts, 'Dataset/Galaxy10_DECals.h5')

    # Crea i DataLoader che gestiranno batching, shuffling e caricamento parallelo
    decals = MakeDataLoaders(opts, data)
    train = decals.train_dataloader
    test = decals.test_dataloader

    LOG.info(f"Dataset caricato: {len(data)} esempi")
    LOG.info(f"Device: {opts.device}")
    LOG.info(f"Batch size: {opts.batch_size}")

    # Avvia il training
    train_loop(model, train, test, opts)


if __name__ == "__main__":
    """
    Entry point del programma:
    1. Carica le configurazioni da Parameters.py
    2. Determina se usare CPU o GPU
    3. Avvia il main con gestione automatica degli errori (ipdb)
    """
    # Converte il dict config in un oggetto per accesso con il punto (es. opts.batch_size)
    opts = SimpleNamespace(**config)

    # Determina automaticamente se usare GPU (cuda) o CPU
    opts.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Esegui il main con debugger automatico in caso di errori
    with launch_ipdb_on_exception():
        main(opts)