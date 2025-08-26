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

    # 1) Load config
    config = load_config(args.config)

    # 2) Generate timestamped folder in ./experiments
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join("experiments", f"run_{timestamp_str}")
    os.makedirs(experiment_dir, exist_ok=True)
    #Add experiment_dir to config
    config["path_directory"] = experiment_dir
    # 3) Save the config inside that folder
    save_config_yaml(config, experiment_dir)
    dm = ScarPINNDataModule(config)
    dm.setup_data()
    min, max = dm.get_minmax_tensors()
    # 4) Instantiate the model
    model = PINN2D(config, min, max)


    # 6) Trainer
    trainer = PINNTrainer(model,dm, config)
    trainer.train()

    # 7) Save final model weights
    model_save_path = os.path.join(experiment_dir, "final_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model and config saved to: {experiment_dir}")

if __name__ == "__main__":
    main()
