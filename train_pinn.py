"""Train a separate PINN for each selected case."""
import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path

from utils import load_config, save_config_yaml
from pinn import PINN2D
from trainer import PINNTrainer
from scar_data_module import ScarPINNDataModule
from case_data import SUPPORTED_CASES, resolve_cases, case_dataset_path


def case_config(config, case):
    selected = deepcopy(config)
    selected.setdefault("DATA", {})["case"] = case
    return selected


def run_case(config, *, check_loader_only=False):
    case = config["DATA"]["case"]
    dm = ScarPINNDataModule(config, case=case)
    dm.setup_data()
    if check_loader_only:
        # Direct indexing avoids a full-grid permutation for this preview.
        batch = dm.dataset[list(range(min(4, len(dm.dataset))))]
        summary = dict(case=case, dataset=str(dm.case_data.path),
                       vertices=dm.case_data.n_vertices,
                       training_samples=len(dm.dataset),
                       batch_shapes={key: list(value.shape) for key, value in batch.items()},
                       feature_order=["x_mm", "y_mm", "time_ms/t_norm", "g_il", "marker"],
                       optimizer_updates=0)
        print(json.dumps(summary, sort_keys=True))
        return summary

    output_root = Path(config["PROJECT"].get("output_root", "experiments"))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    experiment_dir = output_root / case / ("run_fibernet_" + stamp)
    experiment_dir.mkdir(parents=True, exist_ok=False)
    config["PROJECT"]["path_directory"] = str(experiment_dir)
    save_config_yaml(config, experiment_dir)
    mins, maxes = dm.get_minmax_tensors()
    model = PINN2D(config, mins, maxes)
    model.compile()
    trainer = PINNTrainer(model, dm, config)
    if config["TRAINER"]["train_from_checkpoint"]:
        checkpoint_path = str(config["TRAINER"]["checkpoint"]).format(case=case)
        trainer.train_from_checkpoint(checkpoint_path, config["TRAINER"]["phases"])
    else:
        trainer.train()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_multicase.yaml")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--case", choices=SUPPORTED_CASES, help="one case")
    selection.add_argument("--cases", nargs="+", choices=SUPPORTED_CASES,
                           help="cases to process sequentially")
    parser.add_argument("--fibernet-checkpoint-template", help="FiberNet path containing {case}")
    parser.add_argument("--check-loader-only", action="store_true",
                        help="load each case and print batch shapes; no model or training")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.fibernet_checkpoint_template is not None:
        config.setdefault("DATA", {})["fibernet_checkpoint"] = args.fibernet_checkpoint_template
    cases = resolve_cases(config, case=args.case, cases=args.cases)
    paths = [case_dataset_path(case).resolve() for case in cases]
    if len(set(paths)) != len(paths):
        parser.error("Multiple cases must resolve to different HDF5 files")
    if len(cases) > 1 and config["TRAINER"]["train_from_checkpoint"]:
        if "{case}" not in str(config["TRAINER"]["checkpoint"]):
            parser.error("Multi-case continuation requires {case} in TRAINER.checkpoint")
    return [run_case(case_config(config, case), check_loader_only=args.check_loader_only)
            for case in cases]


if __name__ == "__main__":
    main()
