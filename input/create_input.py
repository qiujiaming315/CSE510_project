import numpy as np
import os
import pickle
from pathlib import Path


def save_file(output_path, flow_profile, reprofiling_delay):
    """
    Save the generated input data to the specified output location.
    """
    Path(output_path).mkdir(parents=True, exist_ok=True)
    num_files = len(os.listdir(output_path))
    data = {"flow_profile": flow_profile, "reprofiling_delay": reprofiling_delay}
    with open(f"{output_path}data{num_files + 1}.pickle", 'wb') as f:
        # source, destination
        pickle.dump(data, f)
    return


if __name__ == '__main__':
    flow_profile = [[5, 50, 5], [5, 50, 5], [5, 50, 5], [5, 50, 5], [5, 50, 5], [5, 50, 5], [5, 50, 5], [5, 50, 5],
                    [5, 50, 5], [5, 50, 5]]
    reprofiling_delay = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    save_path = "./data/"
    save_file(save_path, flow_profile, reprofiling_delay)
