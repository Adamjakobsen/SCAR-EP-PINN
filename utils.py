import json
import os
import numpy as np
import h5py
import logging
from datetime import datetime


def get_data():
    # Provide your custom HDF5 read logic or geometry load logic here.
    # This example is just a placeholder.
    # Return (coordinates, triangles, vm):
    with h5py.File("./meshes/scarmesh_1.h5", "r") as scar_file:
        coordinates = scar_file["coordinates"][:]
        triangles = scar_file["topology"][:]
    vm = np.zeros((1, len(coordinates)))  # dummy
    return coordinates, triangles, vm


def get_boundary():
    # Provide logic to load your boundary mesh
    with h5py.File("./meshes/boundary_mesh.h5", "r") as boundary_file:
        group1 = boundary_file["Mesh"]
        group2 = group1["mesh"]
        coordinates = group2["geometry"][:]
        triangles = group2["topology"][:]
    return coordinates, triangles

def load_config(config_path="config.json"):
    with open(config_path, "r") as f:
        return json.load(f)

def make_directory(config):
    path_directory = config.get("path_directory", "./experiments/refactored_run")
    if not os.path.exists(path_directory):
        os.makedirs(path_directory)
    return path_directory


def save_config(config, path):
    with open(os.path.join(path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)


def setup_logger(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s]: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
