import numpy as np
import os
import pickle
from pathlib import Path


def save_file(output_path, flow_profile, arrival_pattern, reprofiling_delay):
    """
    Save the generated input data to the specified output location.
    """
    Path(output_path).mkdir(parents=True, exist_ok=True)
    num_files = len(os.listdir(output_path))
    data = {"flow_profile": flow_profile, "arrival_pattern": arrival_pattern, "reprofiling_delay": reprofiling_delay}
    with open(f"{output_path}data{num_files + 1}.pickle", 'wb') as f:
        # source, destination
        pickle.dump(data, f)
    return


if __name__ == '__main__':
    flow_profile = [[0.5, 5, 5], [0.5, 5, 5], [0.5, 5, 5]]
    pattern1 = [0] * 6
    # pattern1.extend(range(2, 22, 2))
    pattern1.extend(range(2, 52, 2))
    # pattern2 = [1] * 6
    pattern2 = [0] * 6
    # pattern2.extend(range(3, 23, 2))
    pattern2.extend(range(2, 52, 2))
    # pattern3 = [2] * 6
    pattern3 = [0] * 6
    # pattern3.extend(range(4, 24, 2))
    pattern3.extend(range(2, 52, 2))
    arrival_pattern = [pattern1, pattern2, pattern3]
    reprofiling_delay = [5, 5, 5]
    save_path = "./data/"
    save_file(save_path, flow_profile, arrival_pattern, reprofiling_delay)
