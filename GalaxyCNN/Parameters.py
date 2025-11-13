# config.py

config = {
    "model": "simple",
    "augmentation": True,
    "simple_num_filters": 32,
    "simple_mlp_size": 256,

    "seed": 1234,

    "test_size": 0.2,
    "batch_size": 64,
    "learning_rate": 0.0002,
    "weight_decay": 0.0005,
    "num_epochs": 100,

    "rotation_degrees": [0, 180],
    "horizontal_flip_probability": 0.5,
    "vertical_flip_probability": 0.5,

    "log_every": 5,
    "batch_window": 10,
    "save_every": 8,
    "checkpoint_dir": "simple_ckp",
}
