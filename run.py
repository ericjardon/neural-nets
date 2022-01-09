import argparse
import mnist_loader
from shallow_nn import ShallowNetwork
# docs: https://docs.python.org/3/library/argparse.html

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="No desc yet")

    parser.add_argument(
        'layers',
        type=int,
        nargs='+',
        help='Specify the number of neurons layer by layer'
    )

    parser.add_argument(
        "-dataset",
        type=str,
        default="mnist",
        help="Name of the dataset: any from (mnist,...)",
    )

    args = parser.parse_args()
    print("Dataset:", args.dataset)
    print("Layers:", args.layers)
    if args.dataset == 'mnist':
        training, validation, test = mnist_loader.load_data_wrapper()
    net = ShallowNetwork([28*28, 30, 10])

    net.SGD(training, test, epochs=10, mini_batch_size=10, eta=3.0)
    # To speed things up, decrease number of epochs, decrease number of hidden neurons, 
    # or use only a part of the training data.
