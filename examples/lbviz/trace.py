# NOTE: this script is intended to be called from inside the C program!
#       file path resolution may fail otherwise if not called from the proper directory
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import List, Dict
from datetime import datetime

# keys: "name", "start", "end", "track"
Event = Dict[str, float]

# read generated json files from directory into an iterable list
# automatically assigns track indices to display each task's trace in a separate row
def load_events_from_dir(directory: str) -> List[Event]:
    data_dir = Path(directory)
    json_files = sorted(data_dir.glob("*.json"))
    all_events = []
    for track_idx, path in enumerate(json_files):
        with open(path) as f:
            events = json.load(f)
        for ev in events:
            ev["track"] = track_idx
        all_events.extend(events)
    return all_events

# plot the program trace for each (MPI) task
def plot_timeline(dirname, events: List[Event], figsize=(12, None)):
    # build color map from function names
    # (!) later calls overlap earlier calls
    func_names = sorted({ev["name"] for ev in events})
    cmap = plt.get_cmap("tab10") # type: ignore
    color_map = {fn: cmap(i % 10) for i, fn in enumerate(func_names)}
    events_sorted = sorted(events, key=lambda e: (e["start"], -e["end"]))

    # determine height of graph automatically based on number of tasks
    n_tracks = max(ev["track"] for ev in events_sorted) + 1
    height = max(2, n_tracks * 0.5)
    _, ax = plt.subplots(figsize=(figsize[0], height)) # might need figure later?

    # do the actual program trace plot
    # TODO: find a (better) way to also display function names and times in the graph itself
    #       right now, if calls are very close to eachother / very short, the text becomes unreadable
    for ev in events_sorted:
        start, length = ev["start"], ev["end"] - ev["start"]
        track = ev["track"]
        ax.broken_barh([(start, length)], (track * 10, 9), facecolors=color_map[ev["name"]], edgecolors="none", zorder=ev["start"])
        # ax.text(start, track * 10 + 4.5, f"{ev['name']}\n{length:.2f}s", va="center", fontsize=7, color="white", zorder=ev["start"] + 0.1)

    # x axis: (rough) time offset in seconds
    # this is done to improve readability over just printing out the raw unix timestamps
    start_time = min(ev["start"] for ev in events_sorted)
    end_time   = max(ev["end"]   for ev in events_sorted)
    ax.set_xlim(start_time, end_time)
    ax.set_xlabel("Time Offset (s)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{round(x - start_time)}")) # type: ignore

    # y axis: tracks (task ids)
    yticks = [(i * 10 + 4.5) for i in range(n_tracks)] # type: ignore
    ylabels = [str(i) for i in range(n_tracks)] # type: ignore
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_ylabel("Task")
    ax.invert_yaxis() # from low (top) to high (bottom)

    # function legend
    patches = [mpatches.Patch(color=color_map[n], label=n) for n in func_names]
    ax.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc="upper left", title="Function")

    # display
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f'{dirname.split("/json")[-2]}/trace_{timestamp}.svg'
    print(f'saved trace to: {filename}')

    plt.savefig(filename, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    dirname = f'{os.path.dirname(os.path.realpath(__file__))}/json'
    events = load_events_from_dir(dirname)
    plot_timeline(dirname, events)
