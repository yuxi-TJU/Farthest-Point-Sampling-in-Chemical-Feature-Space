from matplotlib import pyplot as plt

import numpy as np
import pickle

from typing import List, Dict
from results import Stats

def plot_scatter(
        prop, stats_list: List[Stats], entity: str, average: bool = False,
    ):
    if stats_list[0].points.get(prop, None) is None:
        raise ValueError(f"Property {prop} not found in stats_list")
    
    scatter_map = {}
    for i, stats in enumerate(stats_list):
        v = stats.points.get(prop).value
        idt = stats.identity
        scatter_map.setdefault(idt, []).append(v)

    if average:
        scatter_map = {k: np.mean(v) for k, v in sorted(scatter_map.items(), key=lambda x: x[0])}
        labels = np.arange(len(scatter_map))
        plt.scatter(labels, list(scatter_map.values()))

    else:
        label_scatter = [len(v) for v in scatter_map.values()]
        labels = np.repeat(np.arange(len(scatter_map)), label_scatter)
        values = np.hstack(list(scatter_map.values()))
        plt.scatter(labels, values)


def plot_stream(
        prop, stats_list: List[Stats], entity: str,
    ):
    if stats_list[0].streams.get(prop, None) is None:
        raise ValueError(f"Property {prop} not found in stats_list")
    
    stream_map = [] # allow for same idt
    for i, stats in enumerate(stats_list):
        sstat = stats.streams.get(prop)
        idt = stats.identity
        stream_map.append((idt, sstat.steps, sstat.values))

    for idt, steps, values in stream_map:
        plt.plot(steps, values, label=idt)
    plt.legend()