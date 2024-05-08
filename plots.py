import numpy as np
import os

from viz.line_error_bar import plot_shaded_error_bar

if __name__ == '__main__':
    # Plot the learning curves.
    path1 = "./output/linear_dqn/reward_history.npz"
    path2 = "./output/deep_dqn/reward_history.npz"
    path3 = "./output/linear_double/reward_history.npz"
    path4 = "./output/deep_double/reward_history.npz"
    learning_reward1 = np.load(path1)
    learning_reward2 = np.load(path2)
    learning_reward3 = np.load(path3)
    learning_reward4 = np.load(path4)
    x_data = [learning_reward1["training_iteration"], learning_reward2["training_iteration"],
              learning_reward3["training_iteration"], learning_reward4["training_iteration"]]
    y_data = [learning_reward1["reward_mean_history"], learning_reward2["reward_mean_history"],
              learning_reward3["reward_mean_history"], learning_reward4["reward_mean_history"]]
    errors = [learning_reward1["reward_std_history"], learning_reward2["reward_std_history"],
              learning_reward3["reward_std_history"], learning_reward4["reward_std_history"]]
    xlabel = "Number of Training Iterations"
    ylabel = "Reward Received"
    output_path = "./figures/"
    os.makedirs(output_path, exist_ok=True)
    fig_name = "learning_curve"
    plot_shaded_error_bar(x_data, y_data, errors, xlabel, ylabel, output_path, fig_name,
                          labels=["Linear", "Deep", "Double Linear", "Double Deep"])
    # TODO: add the code to plot different number of flows.
