import wandb
import os


class FishDetectionModel:
    def __init__(self, args):
        # initialize wandb logging for the project
        wandb.init(project="project-wildlife-ai", entity="adi-ohad-heb-uni")

        if args.dry_run:
            os.environ['WANB_MODE'] = 'dryrun'

        # log all experimental args to wandb
        wandb.config.update(args)

        self.args = args

    def train(self):
        # todo: saving the image and the box in w&b
        # todo: understand the flow of retrieving the data
        # todo: create a training loop and test loop
        print(self.args)

    def load_model(self, path):
        pass

    def predict(self):
        # todo: complete
        pass
