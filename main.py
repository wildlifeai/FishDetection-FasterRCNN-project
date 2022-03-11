import argparse
from train import FishDetectionModel

LEARNING_RATE = 0.001
EPOCHS = 30
BATCH_SIZE = 64
MOMENTOM = 0.0435
WEIGHT_DECAY = 0.00149779494703967
GAMMA = 0.4477536224970189
LEARNING_RATE_SIZE = 4


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=LEARNING_RATE,
        help="learning rate")

    parser.add_argument(
        "-lrs",
        "--learning_rate_size",
        type=int,
        default=LEARNING_RATE,
        help="learning rate size")

    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=EPOCHS,
        help="number of training epochs (passes through full training data)")

    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="batch size of the training")

    parser.add_argument(
        "-m",
        "--momentum",
        type=float,
        default=MOMENTOM,
        help="momentum of the optimizer")

    parser.add_argument(
        "-w",
        "--weight_decay",
        type=float,
        default=MOMENTOM,
        help="weight decay of the optimizer")

    parser.add_argument(
        "-g",
        "--gamma",
        type=float,
        default=MOMENTOM,
        help="weight decay of the optimizer")

    parser.add_argument(
        "-q",
        "--dry_run",
        action="store_true",
        help="Dry run (do not log to wandb)")

    args = parser.parse_args()
    model = FishDetectionModel(args)
    model.train()


if __name__ == '__main__':
    main()
