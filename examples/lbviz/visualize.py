#!/usr/bin/env python3

# NOTE: this script is intended to be called from inside the C program!
#       file path resolution may fail otherwise if not called from the proper directory
import argparse
import csv
import os
import sys
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

# 1. load file, then convert csv rows with ranges to numpy array
def load_array_from_csv_ranges(filename):
    # filename: csv with one of the headers
    # 1d: from_x,to_x,task  (to_x is exclusive)
    # 2d: from_x,to_x,from_y,to_y,task  (to_* exclusive)
    # 3d: from_x,to_x,from_y,to_y,from_z,to_z,task  (to_* exclusive)
    # assumptions: ranges are disjoint, indices are long ints, task numbers are ints, no empty ranges
    ranges = []
    max_indices = []

    with open(filename, newline='') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError('csv has no header')
        
        # case-insensitive header map: lower -> original
        header_map = {h.strip().lower(): h for h in reader.fieldnames}

        # detect format
        has = lambda k: k in header_map
        is_3d = has('from_x') and has('to_x') and has('from_y') and has('to_y') and has('from_z') and has('to_z') and has('task')
        is_2d = has('from_x') and has('to_x') and has('from_y') and has('to_y') and has('task')   and not is_3d
        is_1d = has('from_x') and has('to_x') and has('task')   and not is_2d and not is_3d

        if not (is_1d or is_2d or is_3d):
            raise ValueError('csv must have header for ranges: 1d (from_x,to_x,task) or 2d (from_x,to_x,from_y,to_y,task) or 3d (from_x,to_x,from_y,to_y,from_z,to_z,task)')

        # iterate rows
        f.seek(0)
        reader = csv.DictReader(f)
        for row in reader:
            # skip empty rows
            if not any(v and str(v).strip() for v in row.values()):
                continue

            # parse int helper
            def parse_int(col):
                raw = row.get(header_map[col], '') if col in header_map else row.get(col, '')
                return int(str(raw).strip())

            if is_1d:
                ax = parse_int('from_x'); bx = parse_int('to_x')
                if ax < 0 or bx <= ax:
                    raise ValueError(f'invalid range from_x={ax} to_x={bx} (to_x exclusive and must be > from_x)')
                task = int(str(row[header_map['task']]).strip())
                ranges.append(((ax, bx), task))
                max_indices = [bx] if not max_indices else [max(max_indices[0], bx)]

            elif is_3d:
                ax = parse_int('from_x'); bx = parse_int('to_x')
                ay = parse_int('from_y'); by = parse_int('to_y')
                az = parse_int('from_z'); bz = parse_int('to_z')
                if any(v < 0 for v in (ax, bx, ay, by, az, bz)) or bx <= ax or by <= ay or bz <= az:
                    raise ValueError(f'invalid box ranges x={ax}-{bx}, y={ay}-{by}, z={az}-{bz} (to_* exclusive and must be > froms)')
                task = int(str(row[header_map['task']]).strip())
                ranges.append(((ax, bx, ay, by, az, bz), task))
                if not max_indices:
                    max_indices = [bx, by, bz]
                else:
                    max_indices = [max(max_indices[0], bx), max(max_indices[1], by), max(max_indices[2], bz)]

            else:  # is_2d
                ax = parse_int('from_x'); bx = parse_int('to_x')
                ay = parse_int('from_y'); by = parse_int('to_y')
                if any(v < 0 for v in (ax, bx, ay, by)) or bx <= ax or by <= ay:
                    raise ValueError(f'invalid rectangle ranges x={ax}-{bx}, y={ay}-{by} (to_* exclusive and must be > froms)')
                task = int(str(row[header_map['task']]).strip())
                ranges.append(((ax, bx, ay, by), task))
                max_indices = [bx, by] if not max_indices else [max(max_indices[0], bx), max(max_indices[1], by)]

    # build array using max to_* (exclusive)
    if not max_indices:
        return np.zeros((0,), dtype=np.int64)
    shape = tuple(int(i) for i in max_indices)
    array = np.zeros(shape, dtype=np.int64)

    # apply ranges
    for rng, val in ranges:
        if is_1d:
            a, b = rng
            array[a:b] = int(val)
        elif is_3d:
            ax, bx, ay, by, az, bz = rng
            array[ax:bx, ay:by, az:bz] = int(val)
        else:
            ax, bx, ay, by = rng
            array[ax:bx, ay:by] = int(val)

    return array

# 2. visualize (numpy) array or list using matplotlib, discrete colors, no colorbar
def visualize_array_as_image(array,
                             cmap='gist_rainbow',
                             task_legend=True,
                             out_filename=None):
    array = np.array(array)
    ndim = array.ndim

    # gather unique integer tasks sorted
    unique_vals = np.unique(array).astype(int)
    uniques_sorted = np.sort(unique_vals)
    # mapping from task value -> discrete index 0..N-1
    val_to_idx = {int(v): i for i, v in enumerate(uniques_sorted)}
    N = len(uniques_sorted) if len(uniques_sorted) > 0 else 1
    cmap_obj = plt.cm.get_cmap(cmap, N) # type: ignore

    if ndim == 1:
        # 1d left->right: single row (1, N)
        arr2d = array[np.newaxis, :]
        img = arr2d
        H, W = img.shape
        disp = np.vectorize(lambda v: val_to_idx[int(v)])(img)
        boundaries = np.arange(-0.5, N + 0.5, 1.0)
        norm = mcolors.BoundaryNorm(boundaries, N)
        plt.imshow(disp,
                   origin='lower',
                   extent=[-0.5, W - 0.5, -0.5, H - 0.5],
                   interpolation='nearest',
                   aspect='auto',
                   cmap=cmap_obj,
                   norm=norm)
        plt.gca().set_yticks([])
        plt.ylabel('')

    elif ndim == 2:
        # 2d first dim->y second->x (transpose for imshow so x is horizontal)
        img = array.T
        H, W = img.shape
        disp = np.vectorize(lambda v: val_to_idx[int(v)])(img)
        boundaries = np.arange(-0.5, N + 0.5, 1.0)
        norm = mcolors.BoundaryNorm(boundaries, N)
        plt.imshow(disp,
                   origin='lower',
                   extent=[-0.5, W - 0.5, -0.5, H - 0.5],
                   interpolation='nearest',
                   aspect='auto',
                   cmap=cmap_obj,
                   norm=norm)

    elif ndim == 3:
        # 3d voxel rendering
        nx, ny, nz = array.shape
        mask = np.ones_like(array, dtype=bool)

        # prepare facecolors for each voxel
        facecolors = np.zeros((nx, ny, nz, 4))
        for val, idx in val_to_idx.items():
            norm_idx = idx / (N - 1) if N > 1 else 0.0
            color = cmap_obj(norm_idx)
            facecolors[array == val] = color

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # voxels expects boolean mask ordered x,y,z
        ax.voxels(mask, facecolors=facecolors, edgecolor=None)

        # isometric-ish view
        ax.view_init(elev=30, azim=45)
        try:
            ax.set_box_aspect((nx, ny, nz))
        except Exception:
            pass

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    else:
        raise ValueError('array must be 1d, 2d or 3d')

    # legend mapping discrete colors to integer tasks
    if task_legend:
        patches = []
        for v in uniques_sorted:
            idx = val_to_idx[int(v)]
            norm_idx = idx / (N - 1) if N > 1 else 0.0
            color = cmap_obj(norm_idx)
            patches.append(mpatches.Patch(color=color, label=str(int(v))))
        if ndim == 3:
            ax.legend(handles=patches, title='task', bbox_to_anchor=(1.05, 1), loc='upper left') # type: ignore
            plt.tight_layout(rect=[0, 0, 1, 1])
        else:
            plt.legend(handles=patches, title='task', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout(rect=[0, 0, 1, 1])

    # save to file with current timestamp
    if out_filename is None:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        out_filename = f'plot_{timestamp}.png'
    plt.savefig(out_filename, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='visualize csv ranges (1d/2d/3d)')
    parser.add_argument('filename', help='absolute path to input csv (range form)')
    args = parser.parse_args()

    filename = args.filename
    if not os.path.isabs(filename):
        filename = os.path.abspath(filename)

    if not os.path.exists(filename):
        print(f'error: file not found: {filename}', file=sys.stderr)
        sys.exit(1)

    try:
        array = load_array_from_csv_ranges(filename)
    except Exception as e:
        print(f'error reading csv: {e}', file=sys.stderr)
        sys.exit(1)

    base_dir = os.path.dirname(filename)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out_filename = os.path.join(base_dir, f'plot_{timestamp}.png')

    visualize_array_as_image(array, out_filename=out_filename)
    print(f'saved plot to: {out_filename}')

if __name__ == '__main__':
    main()
