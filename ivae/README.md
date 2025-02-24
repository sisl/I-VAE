# I-VAE
PyTorch implementation of the _inversion variational autoencoder_ (I-VAE).

## Installation
```sh
git clone https://github.com/sisl/I-VAE
cd I-VAE/ivae
pip install .
```

## Usage
```python
from ivae import IVAE, train

model = IVAE(state_channels, obs_channels, latent_dim, dropout_rate)
for epoch in range(epochs):
    train(model, dataloader, val_dataloader, optimizer, device, epoch, epochs)
```

Where `dataloader` holds tuples of `(state, obs)`.

## Test
```sh
python -m unittest discover tests
```
