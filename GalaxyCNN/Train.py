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
    """
    Calcola l'accuracy sul test set.

    - Mette il modello in modalità eval (disattiva dropout, batch norm in training mode)
    - Disabilita il calcolo dei gradienti (torch.no_grad) per risparmiare memoria
    - Per ogni batch: confronta predizioni con labels vere
    - Ritorna l'accuracy media su tutto il test set
    """
    model.eval()  # Modalità valutazione: disattiva dropout, batch norm usa statistiche fisse
    correct = []
    with torch.no_grad():  # Non calcolare gradienti (risparmia memoria e tempo)
        for Xcpu, Y in test:
            X = Xcpu.to(opts.device)
            Y = Y.to(opts.device)
            out = model(X)  # Forward pass: ottieni predizioni

            # Converti output da logits a classi predette (indice della classe con valore massimo)
            pred = np.argmax(N(out), axis=1)

            # Converti one-hot encoding (Y) in indici di classe
            Y_labels = np.argmax(N(Y), axis=1)

            # Confronta predizioni con labels vere
            c = list(pred == Y_labels)
            correct.extend(c)

    # Ritorna percentuale di predizioni corrette
    return np.mean(correct)

#TODO ci mette troppo tempo ad addestrarsi
def train_loop(model, train, test, opts):
    """
    Loop principale di training.

    Per ogni epoca:
    1. Itera su tutti i batch del training set
    2. Per ogni batch:
       - Forward pass: calcola output del modello
       - Calcola loss (CrossEntropyLoss)
       - Backward pass: calcola gradienti
       - Aggiorna pesi con optimizer
    3. Ogni N batch: valuta su test set e logga metriche
    4. Salva checkpoint periodicamente
    5. Aggiorna learning rate con scheduler
    """
    import tensorflow as tf
    # Writer per salvare metriche su TensorBoard (visualizzazione grafici)
    train_writer = tf.summary.create_file_writer(f'tensorboard/{opts.model}/train')
    test_writer = tf.summary.create_file_writer(f'tensorboard/{opts.model}/test')

    # Optimizer: AdamW con weight decay (regolarizzazione L2)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=opts.learning_rate,
                                  weight_decay=opts.weight_decay)

    # Scheduler: riduce learning rate di 10% ogni epoca (gamma=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # Loss function: CrossEntropyLoss (per classificazione multi-classe)
    # Combina Softmax + NegativeLogLikelihood, prende logits (non probabilità)
    loss_function = torch.nn.CrossEntropyLoss()
    step = 0  # Contatore globale per TensorBoard

    for epoch in range(1, opts.num_epochs + 1):
        model.train()  # Modalità training: attiva dropout, batch norm aggiorna statistiche
        mses = []  # Lista delle loss per ogni batch
        correct = []  # Lista delle accuracy per ogni batch
        batch_i = 0

        for Xcpu, Y in train:
            batch_i += 1
            X = Xcpu.to(opts.device)  # Sposta dati su GPU se disponibile
            Y = Y.to(opts.device)

            optimizer.zero_grad()  # Azzera gradienti precedenti
            out = model(X)  # Forward pass: calcola output (logits)

            # Converti Y da one-hot (es. [0,0,1,0,...]) a indice classe (es. 2)
            # CrossEntropyLoss vuole indici, non one-hot
            Y_labels = torch.argmax(Y, dim=1)
            loss = loss_function(out, Y_labels)  # Calcola loss

            loss.backward()  # Backward pass: calcola gradienti
            optimizer.step()  # Aggiorna pesi del modello

            mses.append(N(loss))  # Salva loss per questo batch

            # Calcola accuracy per questo batch
            pred = np.argmax(N(out), axis=1)  # Classe predetta
            true_labels = np.argmax(N(Y), axis=1)  # Classe vera
            c = np.mean(pred == true_labels)  # % predizioni corrette
            correct.append(c)

            # Ogni opts.log_every batch: valuta e logga metriche
            if batch_i % opts.log_every == 0:
                # Media loss e accuracy sugli ultimi batch_window batch
                train_l = np.mean(mses[-opts.batch_window:])
                train_a = np.mean(correct[-opts.batch_window:])

                # Valuta su test set
                test_a = test_metrics(model, test)
                model.train()  # Torna in modalità training

                # Stampa metriche
                msg = f'{epoch:03d}.{batch_i:03d}: '
                msg += f'train: loss={train_l:1.6f}, acc={train_a:1.3f} '
                msg += f'test: acc={test_a:1.3f}'
                LOG.info(msg)

                # Salva metriche su TensorBoard per visualizzazione
                with train_writer.as_default():
                    tf.summary.scalar('loss', train_l, step=step)
                    tf.summary.scalar('accuracy', train_a, step=step)
                with test_writer.as_default():
                    tf.summary.scalar('accuracy', test_a, step=step)
                step += 1

        # Salva checkpoint ogni opts.save_every epoche
        if epoch % opts.save_every == 0:
            save_checkpoint(model, optimizer, epoch, loss, opts)

        # Aggiorna learning rate (lo riduce del 10%)
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