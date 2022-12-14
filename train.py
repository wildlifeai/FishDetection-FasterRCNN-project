import os
import argparse
import wandb
from fish_detection_model import FishDetectionModel

LEARNING_RATE = 1e-5
EPOCHS = 120
BATCH_SIZE = 8
MOMENTUM = 0.0435
WEIGHT_DECAY = 0
GAMMA = 0.1
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
        default=MOMENTUM,
        help="momentum of the optimizer")

    parser.add_argument(
        "-w",
        "--weight_decay",
        type=float,
        default=WEIGHT_DECAY,
        help="weight decay of the optimizer")

    parser.add_argument(
        "-g",
        "--gamma",
        type=float,
        default=GAMMA,
        help="weight decay of the optimizer")

    parser.add_argument(
        "-lm",
        "--load_model",
        type=str,
        default=None,
        help="the path of the trained model")

    parser.add_argument(
        "-mt",
        "--model_type",
        type=object,
        default=None,
        help="Load different model type")

    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default=f"./output/faster-rcnn-model-",
        help="Define an output path for the model")

    parser.add_argument(
        "-dp",
        "--data_path",
        type=str,
        default='./data',
        help="A path to the data folder"
    )

    parser.add_argument(
        "-ver",
        "--verbose",
        type=bool,
        default=True,
        help="Print more if true"
    )

    parser.add_argument(
        "-lck",
        "--load_checkpoint",
        type=bool,
        default=False,
        help="A boolean indicate whether to load checkpoint"
    )

    parser.add_argument(
        "-ckp",
        "--checkpoint_path",
        type=str,
        default='./data/checkpoints_save',
        help="The path to save the checkpoints"
    )

    parser.add_argument(
        "-dr",
        "--dropout",
        type=float,
        default=None,
        help="A dropout to add to the MLPHead"
    )

    parser.add_argument(
        "-sw",
        "--sweep",
        type=int,
        default=4,
        help="A boolean indicate whether to use sweep or single run"
    )

    args = parser.parse_args()
    print(args)

    if not os.path.isdir('./output'):
        os.mkdir("./output")

    if not os.path.isdir('./data/checkpoints_save'):
        os.mkdir("./data/checkpoints_save")

    model = FishDetectionModel(args)

    if args.sweep > 1:
        sweep_config = {
            "name": "wildai-config-sweep",
            "method": "bayes",
            "metric": {
              "name": "total_validation_loss",
              "goal": "minimize"
            },
            "parameters": {
                "epochs": {
                    'distribution': 'int_uniform',
                    'min': 80,
                    'max': 130
                },
                "learning_rate": {
                    'distribution': 'uniform',
                    'min': 0,
                    'max': 1e-4
                },
                "weight_decay": {
                    'distribution': 'uniform',
                    'min': 0,
                    'max': 1e-4
                },
                "learning_rate_size": {
                    'distribution': 'int_uniform',
                    'min': 3,
                    'max': 35
                },
                "gamma": {
                    'distribution': 'uniform',
                    'min': 1e-4,
                    'max': 0.15
                },
            }
        }
        sweep_id = wandb.sweep(sweep_config, project="project-wildlife-ai", entity="adi-ohad-heb-uni")
        wandb.agent(sweep_id, model.train, count=args.sweep)
    else:
        # initialize wandb logging for the project
        wandb.init(project="project-wildlife-ai", entity="adi-ohad-heb-uni")
        model.train()


if __name__ == '__main__':
    main()
