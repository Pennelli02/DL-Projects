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
    Crea e configura un logger con output colorato usando Rich.
    Utile per stampare messaggi formattati durante il training.
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
    Rimuove il tensore dalla GPU e dal grafo computazionale.
    Utile per calcoli con NumPy o per stampare valori.
    """
    return x.detach().cpu().numpy()


def save_checkpoint(model, optimizer, epoch, loss, opts):
    """
    Salva un checkpoint del modello durante il training.

    Salva:
    - Pesi del modello (model.state_dict)
    - Stato dell'optimizer (learning rate, momenti, ecc.)
    - Numero dell'epoca corrente
    - Valore della loss

    Questo permette di riprendere il training in caso di interruzione.
    """
    fname = os.path.join(opts.checkpoint_dir, f'e_{epoch:05d}.chp')
    info = dict(model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                epoch=epoch, loss=loss)
    torch.save(info, fname)
    LOG.info(f'Saved checkpoint {fname}')


def test_metrics(model, test):
    correct = []
    for Xcpu, Y in test:
        X = Xcpu.to(opts.device)
        Y = Y.to(opts.device)
        out = model(X)
        c = list(np.argmax(N(out),axis=1) == np.argmax(N(Y),axis=1))
        correct.extend(c)
    return np.mean(correct)

def train_loop(model, train, test, opts):
    import tensorflow as tf
    train_writer = tf.summary.create_file_writer(f'tensorboard/{opts.model}/train')
    test_writer = tf.summary.create_file_writer(f'tensorboard/{opts.model}/test')
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=opts.learning_rate,
                                  weight_decay=opts.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr=opts.learning_rate)
    loss_function = torch.nn.CrossEntropyLoss()  # This class expects logits
    step = 0
    for epoch in range(1,opts.num_epochs+1):
        model.train()  # Sets the model in training mode
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
            loss = torch.mean(loss)
            loss.backward()
            optimizer.step()
            mses.append(N(loss))
            c = np.mean(np.argmax(N(out),axis=1) == np.argmax(N(Y),axis=1))
            correct.append(c)
            if batch_i % opts.log_every == 0:
                train_l = np.mean(mses[-opts.batch_window:])
                train_a = np.mean(correct[-opts.batch_window:])
                test_a = test_metrics(model, test)
                msg = f'{epoch:03d}.{batch_i:03d}: '
                msg += f'train: loss={train_l:1.6f}, acc={train_a:1.3f} '
                msg += f'test: acc={test_a:1.3f}'
                LOG.info(msg)
                with train_writer.as_default():
                    tf.summary.scalar('loss', train_l, step=step)
                    tf.summary.scalar('accuracy',train_a, step=step)
                with test_writer.as_default():
                    tf.summary.scalar('accuracy',test_a, step=step)
                step += 1
        if epoch % opts.save_every == 0:
            save_checkpoint(model, optimizer, epoch, loss, opts)
        scheduler.step()
        print(scheduler.get_last_lr())


def main(opts):
    """
    Funzione principale: configura e avvia il training.

    1. Crea il modello CNN
    2. Visualizza architettura del modello
    3. Carica dataset (con o senza augmentation)
    4. Crea dataloaders per train/test
    5. Avvia training loop
    """
    from visualizer import visualize
    from GalaxyDataset import GalaxyDataset
    from AugmentedGalaxyDataset import MakeDataLoaders, AugmentedGalaxyDataset

    # Parametri del modello
    input_size = 224  # Dimensione immagini (224x224 pixel)
    input_data = torch.randn(opts.batch_size, 3, input_size, input_size)  # Dati fake per visualizzazione
    num_classes = 10  # 10 tipi di galassie

    # Crea modello CNN
    model = SimpleCNN(num_filters=opts.simple_num_filters,
                      mlp_size=opts.simple_mlp_size,
                      num_classes=num_classes)

    # Visualizza architettura (salva grafo del modello)
    visualize(model, 'myCNN', input_data)

    # Sposta modello su GPU se disponibile
    model = model.to(opts.device)

    # Carica dataset: con augmentation (rotazioni, flip) o senza
    if opts.augmentation:
        data = AugmentedGalaxyDataset(opts, 'Dataset/Galaxy10_DECals.h5')
    else:
        data = GalaxyDataset(opts, 'Dataset/Galaxy10_DECals.h5')

    # Crea dataloaders: gestiscono batch, shuffle, split train/test
    decals = MakeDataLoaders(opts, data)
    train = decals.train_dataloader
    test = decals.test_dataloader

    # Avvia training
    train_loop(model, train, test, opts)


if __name__ == "__main__":
    """
    Punto di ingresso del programma.

    1. Carica configurazione da Parameters.py
    2. Converte dict in oggetto (per accedere con opts.batch_size invece di opts['batch_size'])
    3. Rileva se GPU disponibile
    4. Avvia main con debugger automatico in caso di errori
    """
    # Converte dict config in oggetto SimpleNamespace
    opts = SimpleNamespace(**config)

    # Usa GPU se disponibile, altrimenti CPU
    opts.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # launch_ipdb_on_exception: se c'è un errore, apre debugger invece di crashare
    with launch_ipdb_on_exception():
        main(opts)