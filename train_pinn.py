import torch
import argparse
import os
from datetime import datetime

from utils import load_config, save_config_yaml
from pinn import PINN2D
from trainer import PINNTrainer
from scar_data_module import ScarPINNDataModule
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Generate timestamped folder in ./experiments
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    experiment_dir = os.path.join("experiments", f"run_fibernet_{timestamp_str}")
    os.makedirs(experiment_dir, exist_ok=True)
    #Add experiment_dir to config
    config["PROJECT"]["path_directory"] = experiment_dir
    # Save the config inside that folder
    save_config_yaml(config, experiment_dir)
    dm = ScarPINNDataModule(config)
    if config["MODEL"]["sigmanet"]:
        dm.setup_data()
    else:
        dm.setup_data_with_conductivities()
    min, max = dm.get_minmax_tensors()
    # Instantiate the model
    model = PINN2D(config, min, max)
    model.compile()


    # Trainer
    trainer = PINNTrainer(model,dm, config)
    if config["TRAINER"]["train_from_checkpoint"]:
        checkpoint_path = config["TRAINER"]["checkpoint"]
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        trainer.train_from_checkpoint(checkpoint_path,config["TRAINER"]["phases"])
    else:
        trainer.train()



if __name__ == "__main__":
    main()
