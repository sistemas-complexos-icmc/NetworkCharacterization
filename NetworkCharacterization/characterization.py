import networkx as nx
import numpy as np
import pandas as pd
import igraph as ig
from dataclasses import dataclass
from tqdm import tqdm
import os
import math

class NetworkCharacterization:
    igraph: ig.GraphBase
    nxgraph: nx.Graph
    measure_dict: dict[str, float]
    degree_values: np.array
    degree_probability: np.array

    def __init__(self, network_file: str):
        self.igraph = ig.Graph.Read_Edgelist(network_file, directed=False)
        self.nxgraph = nx.Graph(self.igraph.get_edgelist())

    def get_measure_dict(self) -> dict[str, float]:
        measure_dict = {
            'clustering': self.clustering({'mode': 'zero'}),
            'closeness': self.closeness({}),
            'size': self.size({}),
            'betweenness': self.betweenness({}),
            'average_shortest_path_lenght': self.average_shortest_path_lenght({}),
            'eigenvector': self.eigenvector({}),
            'assortativity': self.assortativity({'directed': False}),
            'information_centrality': self.information_centrality({}),
            'approximate_current_flow_betweenness_centrality': self.approximate_current_flow_betweenness_centrality({}),
            'shannon_entropy': self.shannon_entropy({}),
            'degree_variance': self.degree_variance({})
        }
        return measure_dict

    def clustering(self, kwargs: dict | None = None) -> float:
        return self.igraph.transitivity_undirected(**kwargs)

    def closeness(self, kwargs: dict | None = None) -> float:
        return np.mean(self.igraph.closeness(**kwargs))

    def betweenness(self, kwargs: dict | None = None) -> float:
        return np.mean(self.igraph.betweenness(**kwargs))

    def average_shortest_path_lenght(self, kwargs: dict | None = None) -> float:
        return np.mean(self.igraph.distances(**kwargs))
    
    def eigenvector(self, kwargs: dict | None = None) -> float:
        return np.mean(self.igraph.eigenvector_centrality(**kwargs))

    def assortativity(self, kwargs: dict | None = None) -> float:
        return self.igraph.assortativity_degree(**kwargs)

    def information_centrality(self, kwargs: dict | None = None) -> float:
        return np.mean(list(nx.information_centrality(self.nxgraph, **kwargs).values()))

    def approximate_current_flow_betweenness_centrality(self, kwargs: dict | None = None) -> float:
        return np.mean(list(nx.approximate_current_flow_betweenness_centrality(self.nxgraph, **kwargs).values()))

    def shannon_entropy(self, kwargs: dict | None = None) -> float:
        k, Pk = self.degree_distribution()
        H = 0
        for p in Pk:
            if (p > 0):
                H -= p * math.log(p, 2)
        return H

    def degree_variance(self, kwargs: dict | None = None) -> float:
        k, Pk = self.degree_distribution()
        return np.std(k)

    def degree_distribution(self):
        degree_list = np.array(list(dict(self.nxgraph.degree()).values()))
        max_degree = np.max(degree_list)
        degree_values = np.arange(0, max_degree + 1)
        degree_probability = np.zeros(max_degree + 1)

        for k in degree_list:
            degree_probability[k] += 1

        degree_probability = degree_probability/sum(degree_probability)
        return degree_values, degree_probability

    def size(self, kwargs: dict | None = None):
        return len(self.nxgraph)


def compute_network_measures(network_directory: str) -> pd.DataFrame:
    measure_dict_df = {}
    for network_file in tqdm(os.listdir(network_directory)):
        network_measure = NetworkCharacterization(f'{network_directory}/{network_file}')
        measure_dict = network_measure.get_measure_dict()
        measure_dict_df[network_file] = measure_dict
    measure_dict_df = pd.DataFrame(measure_dict_df).T
    return measure_dict_df


if __name__ == '__main__':
    df = compute_network_measures('barabasi_linear')
    print(df.head())