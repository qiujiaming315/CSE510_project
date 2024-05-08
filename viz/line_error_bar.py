import os
from matplotlib import pyplot as plt


def plot_shaded_error_bar(x_data, y_data, errors, xlabel, ylabel, output_path, fig_name, labels=[]):
    plt.rc("font", family="DejaVu Sans")
    plt.rcParams['figure.figsize'] = (15, 10)
    ax0 = plt.subplot()
    ax0.tick_params(axis='x', labelsize=30)
    ax0.tick_params(axis='y', labelsize=30)
    ax0.spines['top'].set_color('#606060')
    ax0.spines['bottom'].set_color('#606060')
    ax0.spines['left'].set_color('#606060')
    ax0.spines['right'].set_color('#606060')
    ax0.grid(True, color='#bfbfbf', linewidth=1)
    ax0.set_ylabel(ylabel, labelpad=10, color='#333333', size=40)
    ax0.set_xlabel(xlabel, labelpad=15, color='#333333', size=40)
    colors = ["#7FB3D5", "#F7CAC9", "#A2C8B5", "#D9AFD9"]
    assert len(colors) >= len(x_data), "Too many lines to plot."
    legend = len(labels) > 0
    if not legend:
        labels = [""] * len(x_data)
    for x, y, error, color, label in zip(x_data, y_data, errors, colors, labels):
        ax0.plot(x, y, '-', color=color, label=label, linewidth=3, markersize=12)
        ax0.fill_between(x, y - error, y + error, color=color, alpha=0.5)
    if legend:
        plt.legend(fontsize=35)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, fig_name + ".png"), bbox_inches='tight')
