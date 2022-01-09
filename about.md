# MNIST DATASET
```
pip install numpy  # fast linear algebra

git clone https://github.com/mnielsen/neural-ShallowNetworks-and-deep-learning.git
```

## Loading the MNIST Dataset
`mnist_loader.py` is a helper program.

This project is meant to run several types of nerual nets with
different datasets and different shapes for the neural net.

The entrypoint is the `run.py` script, which currently accepts
the following command line arguments:
- `layers` required, indicates the number of neurons layer by layer
- `-dataset` optional, indicates the name of the dataset.

**Supported datasets (09-01-2021):**
- MNIST digit classification dataset (60,000), gz compressed  pickle file

````
>>> import mnist_loader
>>> training_data , validation_data, test_data = mnist_loader.load_data_wrapper()
```

