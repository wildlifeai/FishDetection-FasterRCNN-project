import wandb
import os


class FishDetectionModel:
    def __init__(self, args):
        # initialize wandb logging for the project
        wandb.init(project="project-wildlife-ai", entity="adi-ohad-heb-uni")

        if args.dry_run:
            print("I am hereeee")
            os.environ['WANB_MODE'] = 'dryrun'

        # log all experimental args to wandb
        wandb.config.update(args)

        self.args = args

    def train(self):
        print(self.args)

    def load(self, path):
        pass
