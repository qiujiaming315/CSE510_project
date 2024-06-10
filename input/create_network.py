import numpy as np
import os
import pickle
from pathlib import Path


def save_file(output_path, flow_path):
    """
    Save the generated input data to the specified output location.
    """
    Path(output_path).mkdir(parents=True, exist_ok=True)
    num_files = len(os.listdir(output_path))
    np.save(f"{output_path}data{num_files + 1}.npy", flow_path)
    return


if __name__ == '__main__':
    # flow_path = [[1], [1], [1]]
    # flow_path = [[1, 2, 0, 0], [0, 1, 2, 0], [0, 0, 1, 2]]
    # flow_path = [[1, 0, 2, 0, 0, 3, 0, 4], [0, 1, 0, 2, 0, 0, 3, 4], [0, 0, 0, 0, 1, 0, 2, 3]]
    flow_path = [[1, 2, 0], [0, 1, 2], [2, 0, 1]]
    save_path = f"./data/flow_path/flow{len(flow_path)}/"
    save_file(save_path, flow_path)
