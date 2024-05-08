import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick


def plot_delay(arrival_time, end_to_end_delay, output_path, fig_name, segment=None):
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
    ax0.set_ylabel("End-to-end Delay", labelpad=10, color='#333333', size=40)
    ax0.set_xlabel("Packet Arrival Time", labelpad=15, color='#333333', size=40)
    ax0.yaxis.set_major_formatter(mtick.PercentFormatter())
    num_flow = len(arrival_time)
    colors = ["#7FB3D5", "#F7CAC9", "#A2C8B5", "#D9AFD9", "#D3D3D3", "#FFFF99", "#FFD1DC", "#9AD1D4", "#B19CD9",
              "#B0AFAF"]
    assert len(colors) >= num_flow, "Too many flows."
    labels = [f"flow {i + 1}" for i in range(num_flow)]
    arrival_aggregate = []
    for xdata, ydata, color, label in zip(arrival_time, end_to_end_delay, colors, labels):
        xdata, ydata = np.array(xdata), np.array(ydata)
        if segment is not None:
            mask = np.logical_and(xdata >= segment[0], xdata <= segment[1])
            xdata, ydata = xdata[mask], ydata[mask]
        ax0.plot(xdata, ydata, 'o-', color=color, label=label, linewidth=3, markersize=9)
        arrival_aggregate.extend(list(xdata))
    ax0.hlines(100, np.amin(arrival_aggregate), np.amax(arrival_aggregate), colors="red", linewidth=5,
               label='hard delay bound')
    plt.legend(fontsize=35)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, fig_name + ".png"), bbox_inches='tight')
    plt.clf()
