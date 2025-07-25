# NOTE: this script is intended to be called from inside the C program!
#       file path resolution may fail otherwise if not called from the proper directory
import ast
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from datetime import datetime

PREFIX = 'lbviz/' # needed if the script is called automatically through the lb program / C code

# 1. load file, then convert tuple structure to numpy array
def load_array_from_tuples(filename):
    data = []
    max_indices = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx_val = ast.literal_eval(line)
            idx, val = idx_val
            data.append((idx, val))
            if isinstance(idx, tuple):
                if not max_indices:
                    max_indices = list(idx)
                else:
                    max_indices = [max(max_indices[d], idx[d]) for d in range(len(idx))]
            else:
                if not max_indices:
                    max_indices = [idx]
                else:
                    max_indices[0] = max(max_indices[0], idx)

    shape = tuple(i + 1 for i in max_indices)
    array = np.zeros(shape)
    for idx, val in data:
        array[idx] = val

    return array

# 2. visualize (numpy) array or list using matplotlib
def visualize_array_as_image(array,
                             cmap='gist_rainbow',
                             show_colorbar=True,
                             orientation='horizontal',
                             task_legend=True):
    array = np.array(array)

    # if 1D, promote to 2D (single row)
    if array.ndim == 1:
        array = array[np.newaxis, :]
        plt.gca().set_yticks([])
        plt.ylabel('')

    # transpose so first dim→y, second→x
    img = array.T
    H, W = img.shape

    # plot with origin in lower-left and proper pixel alignment
    plt.imshow(img,
               origin='lower',
               extent=[-0.5, W-0.5, -0.5, H-0.5],
               interpolation='nearest',
               aspect='auto',
               cmap=cmap)

    # optional task legend
    if task_legend:
        unique_vals = np.unique(img)
        norm = mcolors.Normalize(vmin=unique_vals.min(), vmax=unique_vals.max())
        cmap_obj = plt.cm.get_cmap(cmap)
        patches = []
        for val in unique_vals:
            color = cmap_obj(norm(val))
            patches.append(mpatches.Patch(color=color, label=str(int(val))))
        plt.legend(handles=patches,
                   title='Task',
                   bbox_to_anchor=(1.05, 1),
                   loc='upper left')
        plt.tight_layout(rect=[0, 0, 1, 1])

    # save
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f'{PREFIX}plot_{timestamp}.png'
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


# 1. + 2.
def main():
    filename = f'{PREFIX}array_data.txt' 
    array = load_array_from_tuples(filename)
    visualize_array_as_image(array)

if __name__ == '__main__':
    main()