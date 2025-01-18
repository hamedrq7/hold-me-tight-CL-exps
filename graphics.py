import matplotlib
import matplotlib.pyplot as plt
import numpy as np

W = 10
H = 0.4 * W

matplotlib.rc('text')
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False

FONT_SIZE = 15
MARKER_SIZE = 15
LINEWIDTH = 2
SAMPLE_SIZE = 3

plt.rc('font', size=FONT_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=FONT_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=FONT_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=FONT_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=FONT_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=FONT_SIZE)    # legend fontsize
plt.rc('figure', titlesize=FONT_SIZE)  # fontsize of the figure title

def swarmplot(margins, name: str, alpha=0.3, jitter=0.1, color='red', s=SAMPLE_SIZE, discard_ratio=0):
    plt.figure(figsize=(W, H))
    idx = np.tile(np.arange(margins.shape[0]), (margins.shape[1], 1)).T
    plt.plot(idx[:,0], np.median(margins, axis=1), '.-', linewidth=LINEWIDTH, color=color, markersize=MARKER_SIZE)
    idx = idx[:, :int(margins.shape[1] * (1-discard_ratio))]
    max_median = np.median(margins, axis=1).max()
    margins = margins[:,:int(margins.shape[1] * (1-discard_ratio))]
    plt.scatter(idx[:]+ jitter * np.random.randn(*margins.shape), margins[:], alpha=alpha, color=color, s=s)
    
    new_xticks=['Low', 'Frequency', 'High']
    plt.xticks([-1, 10, 21], new_xticks, rotation=0, horizontalalignment='center')
    plt.axis([-1, 21, 0, max_median * 1.3])
    # plt.axis([-1, 21, 0., 15.])
    plt.ylabel('Margin')

    plt.savefig(f'{name}.png')

def swarmplot_list(margin_list, labels, alpha=0.3, jitter=0.1, colors=None, s=SAMPLE_SIZE, discard_ratio=0):
    """
    Plot multiple arrays of margins on the same figure.
    
    Args:
    - margin_list: List of numpy arrays, where each array contains margin data to be plotted.
    - labels: List of labels corresponding to each margin array.
    - alpha: Transparency level for scatter points.
    - jitter: Jitter added to the scatter points for better visualization.
    - colors: List of colors for each margin array.
    - s: Size of scatter points.
    - discard_ratio: Ratio of points to discard for each margin array.
    """
    plt.figure(figsize=(W, H))
    
    if colors is None:
        colors = ['red', 'blue', 'green', 'orange', 'purple']  # Default colors
    
    max_median = 0
    for i, margins in enumerate(margin_list):
        idx = np.tile(np.arange(margins.shape[0]), (margins.shape[1], 1)).T
        # Plot median line
        plt.plot(idx[:, 0], np.median(margins, axis=1), '.-', linewidth=LINEWIDTH, 
                 color=colors[i % len(colors)], markersize=MARKER_SIZE, label=labels[i])
        
        # Update max_median for axis limits
        max_median = max(max_median, np.median(margins, axis=1).max())
        
        # Plot scatter points
        margins = margins[:, :int(margins.shape[1] * (1 - discard_ratio))]
        idx = idx[:, :int(margins.shape[1])]
        plt.scatter(idx[:] + jitter * np.random.randn(*margins.shape), margins[:], 
                    alpha=alpha, color=colors[i % len(colors)], s=s)
    
    new_xticks = ['Low', 'Frequency', 'High']
    plt.xticks([-1, 10, 21], new_xticks, rotation=0, horizontalalignment='center')
    plt.axis([-1, 21, 0, max_median * 1.3])
    plt.ylabel('Margin')
    plt.legend()
    plt.show()

from typing import List 
import seaborn as sns

def histogram_coupled(arrays: List, labels: List): 
    plt.figure(figsize=(10, 6))

    for index in range(len(arrays)):
        sns.kdeplot(
            arrays[index],
            label=f'{labels[index]} (Mean: {np.mean(arrays[index]):.2f})',
            linewidth=2,
            clip=(0, None)  # Restrict KDE to values >= 0
        )

    plt.xlabel('l-2')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__": 
    # margins1 = np.load('/home/ramin/Margin_analysis/hold-me-tight-CL-exps/Models/Generated/MNIST/LeNet-tradesSetting_True/CE/margins.npy')
    # margins2 = np.load('/home/ramin/Margin_analysis/hold-me-tight-CL-exps/Models/Generated/MNIST/LeNet-tradesSetting_True/CL/center_lr-0.5 alpha-0.1/margins.npy')
    
    margins1 = np.load('/home/ramin/Margin_analysis/hold-me-tight-CL-exps/Models/Generated/MNIST/MNIST_TRADES-tradesSetting_False/CE/margins.npy')
    margins2 = np.load('/home/ramin/Margin_analysis/hold-me-tight-CL-exps/Models/Generated/MNIST/MNIST_TRADES-tradesSetting_False/CL/center_lr-0.5 alpha-0.1 epochs-30/margins.npy')

    # margins1 = np.load('/home/ramin/Margin_analysis/hold-me-tight-CL-exps/Models/Generated/CIFAR10/ResNet18/CE/margins.npy')
    # margins2 = np.load('/home/ramin/Margin_analysis/hold-me-tight-CL-exps/Models/Generated/CIFAR10/ResNet18/CL/center_lr-0.5 alpha-0.01 epochs-30/margins.npy')

    # margins1 = np.load('/home/ramin/Margin_analysis/hold-me-tight-CL-exps/Models/Generated/CIFAR10/ResNet18/AT_CE/margins.npy')
    # margins2 = np.load('/home/ramin/Margin_analysis/hold-me-tight-CL-exps/Models/Generated/CIFAR10/ResNet18/AT_CLEAN_CENT/margins.npy')

    # margins1 = np.load('/home/ramin/Margin_analysis/hold-me-tight-CL-exps/Models/Generated/CIFAR10/ResNet18/CE/margins.npy')
    # margins2 = np.load('/home/ramin/Margin_analysis/hold-me-tight-CL-exps/Models/Generated/CIFAR10/ResNet18/CL/center_lr-0.5 alpha-0.01 epochs-30/margins.npy')

    
    name1 = 'CE'
    name2 = 'CL'
    swarmplot_list([margins1, margins2], labels=[name1, name2])

    margins1 = np.load('/home/ramin/Margin_analysis/hold-me-tight-CL-exps/Models/Generated/CIFAR10/ResNet18/CE/l2_norms.npy')
    margins2 = np.load('/home/ramin/Margin_analysis/hold-me-tight-CL-exps/Models/Generated/CIFAR10/ResNet18/CL/center_lr-0.5 alpha-0.01 epochs-30/l2_norms.npy')

    histogram_coupled([margins1, margins2], ['CE', 'CL'])