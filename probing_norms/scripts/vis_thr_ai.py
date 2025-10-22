import json
import pdb
import random
import os
import pickle

from collections import defaultdict
from dataclasses import dataclass
from itertools import groupby
from functools import partial
from typing import Dict, List

import click
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, RepeatedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, Normalizer

from tqdm import tqdm

from probing_norms.constants import NUM_MIN_CONCEPTS
from probing_norms.data import (
    DATASETS,
    DIR_LOCAL,
    filter_by_things_concepts,
    load_features_metadata,
    load_binder_dense,
    load_binder_feature_norms,
    load_binder_feature_norms_median,
    load_mcrae_feature_norms,
    load_mcrae_x_things,
    get_feature_to_concepts,
)
from probing_norms.utils import cache_json, implies, read_file
from probing_norms.extract_features_image import (
    FEATURE_EXTRACTORS as FEATURE_EXTRACTORS_IMAGE,
)
from probing_norms.extract_features_text import (
    FEATURE_EXTRACTORS as FEATURE_EXTRACTORS_TEXT,
    MAPPING_TYPES,
)


NORMS_PRIMING = "mcrae"
NORMS_NUM_RUNS = 30


def get_idxs_similar():
    with open('./probing_norms/similar_idxs.txt') as f:
        numbers = [int(x) for x in f.read().split()]
        return numbers

def aggregate_by_labels(embeddings, labels):
    unique_labels = np.unique(labels)
    agg_embeddings = np.zeros((len(unique_labels), embeddings.shape[1]))
    for i, label in enumerate(unique_labels):
        idxs = labels == label
        agg_embeddings[i] = embeddings[idxs].mean(axis=0)
    return agg_embeddings, unique_labels


AGGREGATE_EMBEDDINGS = {
    "instance": lambda x, y: (x, y),
    "concept": aggregate_by_labels,
}


FEATURE_TYPE_TO_MODALITY = {
    **{k: "image" for k in FEATURE_EXTRACTORS_IMAGE.keys()},
    **{
        k + "-" + m: "text"
        for k in FEATURE_EXTRACTORS_TEXT.keys()
        for m in MAPPING_TYPES
    },
}

import math

def community_layout_radial(G, partition, *, radius=10.0, cluster_spread=0.5, seed=42):
    rng = np.random.default_rng(seed)

    # group nodes by community
    clusters = {}
    for n, cl in partition.items():
        clusters.setdefault(cl, []).append(n)

    num_clusters = len(clusters)
    pos = {}

    for i, (cl, nodes) in enumerate(clusters.items()):
        angle = 2 * math.pi * i / num_clusters
        cx, cy = radius * math.cos(angle), radius * math.sin(angle)

        sub = G.subgraph(nodes)
        k_local = max(0.05, cluster_spread / math.sqrt(max(1, sub.number_of_nodes())))
        local = nx.spring_layout(
            sub,
            seed=int(rng.integers(0, 1_000_000)),
            k=k_local,
            scale=cluster_spread,
        )
        for n, (x, y) in local.items():
            pos[n] = (x + cx, y + cy)

    return pos

def load_embeddings(dataset_name, feature_type, embeddings_level):
    modality = FEATURE_TYPE_TO_MODALITY[feature_type]
    path = "/root/output/features-{}/{}-{}.npz".format(modality, dataset_name, feature_type)
    output = np.load(path, allow_pickle=True)
    embeddings = output["X"]
    labels = output["y"].astype(np.int32)
    if modality == "image":
        embeddings, labels = AGGREGATE_EMBEDDINGS[embeddings_level](embeddings, labels)
    return embeddings, labels


def predict_func_classifier(clf, X_te):
    return clf.predict_proba(X_te)[:, 1]


def predict_func_regression(clf, X_te):
    return clf.predict(X_te)


PREDICT_FUNCS = defaultdict(lambda: predict_func_classifier)
PREDICT_FUNCS["LinearRegression"] = predict_func_regression


def predict1(make_classifier, X, y, split):
    idxs_tr = split.tr_idxs
    idxs_te = split.te_idxs

    X_tr, y_tr = X[idxs_tr], y[idxs_tr]
    X_te, y_te = X[idxs_te], y[idxs_te]

    clf = make_classifier()
    predict_func = PREDICT_FUNCS[clf.__class__.__name__]

    clf.fit(X_tr, y_tr)
    y_pr = predict_func(clf, X_te)

    preds = [
        {
            "i": i.item(),
            # "name": dataset.image_files[i],
            # "label": dataset.labels[i],
            "pred": p.item(),
            "true": t.item(),
        }
        for i, p, t in zip(idxs_te, y_pr, y_te)
    ]
    return {
        "preds": preds,
        "clf": clf,
    }


def predict_splits(make_classifier, X, y, splits):
    return [
        {
            "split": split.metadata,
            **predict1(make_classifier, X, y, split),
        }
        for split in tqdm(splits, leave=False)
    ]


@dataclass
class Split:
    tr_idxs: np.ndarray
    te_idxs: np.ndarray
    metadata: dict

def exclude_from_test(split:Split,exclude : List[int]) -> Split :
    s = set(exclude)
    
    s_train= set(split.tr_idxs)
    s_test = set(split.te_idxs)

    s_train |= s 
    s_test -= s 
    return Split(np.array(list(s_train)),np.array(list(s_test)),exclude.metadata)

def get_train_test_split_iid_fixed(
    labels,
    features,
    **_,
) -> Dict[str, List[Split]]:
    forced_in_train = get_idxs_similar()
    idxs = np.arange(len(labels))
    idxss = train_test_split(idxs, test_size=0.2, random_state=42)
    get_f = lambda f: [exclude_from_test(Split(*idxss, metadata=dict(feature=f)),forced_in_train)]
    return {feature: get_f(feature) for feature in features}


def get_train_test_split_leave_one_out(
    labels,
    features,
    *,
    feature_to_concepts,
    class_to_label,
) -> Dict[str, List[Split]]:
    def get_c(concept):
        forced_in_train = get_idxs_similar()
        idxs = list(range(len(labels)))
        label = class_to_label[concept]
        tr_idxs = [i for i in idxs if labels[i] != label]
        te_idxs = [i for i in idxs if labels[i] == label]
        return {
            "tr_idxs": np.array(tr_idxs),
            "te_idxs": np.array(te_idxs),
        }

    def get_f(feature):
        concepts = feature_to_concepts[feature]
        return [
            exclude_from_test(Split(
                **get_c(concept),
                metadata={"feature": feature, "test-concept": concept},
            ),forced_in_train)
            for concept in concepts
        ]

    return {feature: get_f(feature) for feature in features}


def get_binary_labels(labels, feature, feature_to_concepts, class_to_label):
    concepts_str = feature_to_concepts[feature]
    concepts_num = [class_to_label[c] for c in concepts_str]
    concepts_num = set(concepts_num)
    binary_labels = [label in concepts_num for label in labels]
    return np.array(binary_labels).astype(int)


def get_continuous_labels(labels, feature, feature_to_concepts, class_to_label):
    concepts_and_scores = feature_to_concepts[feature]
    concept_to_score = {class_to_label[c]: s for c, s in concepts_and_scores}
    labels_out = [concept_to_score[label] for label in labels]
    return np.array(labels_out)

def get_train_test_split_k_fold(
    labels,
    features,
    *,
    feature_to_concepts,
    class_to_label,
    n_splits=5,
    n_repeats=2,
):
    def get_f(feature):
        binary_labels = get_binary_labels(
            labels,
            feature,
            feature_to_concepts,
            class_to_label,
        )
        rskf = RepeatedStratifiedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=42,
        )
        return [
            Split(
                *idxss,
                metadata={"feature": feature},
            )
            for idxss in rskf.split(binary_labels, binary_labels)
        ]

    return {feature: get_f(feature) for feature in features}


def get_train_test_split_k_fold_simple(
    labels,
    features,
    *,
    feature_to_concepts,
    class_to_label,
    n_splits=5,
    n_repeats=2,
):
    def get_f(feature):
        indices = np.arange(len(labels))
        rskf = RepeatedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=42,
        )
        return [
            Split(
                *idxss,
                metadata={"feature": feature},
            )
            for idxss in rskf.split(indices)
        ]

    return {feature: get_f(feature) for feature in features}


GET_TRAIN_TEST_SPLIT = {
    "iid-fixed": get_train_test_split_iid_fixed,
    "leave-one-concept-out": get_train_test_split_leave_one_out,
    "repeated-k-fold": get_train_test_split_k_fold,
    "repeated-k-fold-simple": get_train_test_split_k_fold_simple,
}


def sample_features(feature_to_concepts):
    features_selected = [
        feature
        for feature, concepts in feature_to_concepts.items()
        if len(concepts) >= NUM_MIN_CONCEPTS
    ]

    random.seed(42)
    features_selected_1 = random.sample(features_selected, 64)
    features_selected_2 = random.sample(features_selected, 256 - 64)
    features_selected = sorted(set(features_selected_1 + features_selected_2))
    return features_selected


class NormsLoader:
    def __call__(self):
        raise NotImplementedError

    def get_suffix(self) -> str:
        raise NotImplementedError

    def load_concepts(self):
        path = "data/concepts-things.txt"
        path = DIR_LOCAL / path
        return read_file(path)


class GPT3NormsLoader(NormsLoader):
    def __init__(self, norms_model):
        self.norms_model = norms_model

    def __call__(self):
        feature_to_concepts, feature_to_id = load_features_metadata(
            priming=NORMS_PRIMING,
            model=self.norms_model,
            num_runs=NORMS_NUM_RUNS,
        )
        features_selected = sample_features(feature_to_concepts)
        return feature_to_concepts, feature_to_id, features_selected

    def get_suffix(self):
        return "{}_{}_{}".format(NORMS_PRIMING, self.norms_model, NORMS_NUM_RUNS)


class McRaeNormsLoader(NormsLoader):
    def __init__(self):
        self.model = "mcrae"

    def __call__(self, *, num_min_concepts=NUM_MIN_CONCEPTS):
        concept_feature = load_mcrae_feature_norms()
        feature_to_concepts = get_feature_to_concepts(concept_feature)

        features = sorted(feature_to_concepts.keys())
        feature_to_id = {feature: i for i, feature in enumerate(features)}

        features_selected = [
            feature
            for feature, concepts in feature_to_concepts.items()
            if len(concepts) >= num_min_concepts
        ]
        return feature_to_concepts, feature_to_id, features_selected

    def get_suffix(self):
        return str(self.model)


class McRaeMappedNormsLoader(NormsLoader):
    def __init__(self):
        self.model = "mcrae-to-gpt35"

    def __call__(self, *, num_min_concepts=10):
        path = "data/mcrae++.json"
        path = DIR_LOCAL / path
        with open(path) as f:
            data = json.load(f)

        feature_to_concepts = {d["norm"]: d["concepts"] for d in data}
        feature_to_id = {d["norm"]: i for i, d in enumerate(data)}
        features_selected = [
            norm
            for norm, concepts in feature_to_concepts.items()
            if len(concepts) >= num_min_concepts
        ]
        return feature_to_concepts, feature_to_id, features_selected

    def get_suffix(self):
        return str(self.model)


class McRaeXThingsNormsLoader(NormsLoader):
    def __init__(self):
        self.model = "mcrae-x-things"

    def __call__(self, *, num_min_concepts=10):
        concept_feature = cache_json("data/mcrae-x-things.json", load_mcrae_x_things)
        feature_to_concepts = get_feature_to_concepts(concept_feature)

        features = sorted(feature_to_concepts.keys())
        feature_to_id = {feature: i for i, feature in enumerate(features)}

        features_selected = [
            norm
            for norm, concepts in feature_to_concepts.items()
            if len(concepts) >= num_min_concepts
        ]
        return feature_to_concepts, feature_to_id, features_selected

    def get_suffix(self):
        return str(self.model)


class BinderNormsLoader(NormsLoader):
    def __init__(self, thresh):
        self.model = "binder"
        self.thresh = thresh
        if isinstance(thresh, str):
            assert thresh == "median"
            self.load_feature_norms = load_binder_feature_norms_median
        elif isinstance(thresh, int):
            self.load_feature_norms = partial(load_binder_feature_norms, thresh=thresh)
        else:
            assert False

    def load_concepts(self):
        path = "data/binder-norms.xlsx"
        path = DIR_LOCAL / path
        df = pd.read_excel(path)
        df = filter_by_things_concepts(df)
        return df["Word"].tolist()

    def __call__(self):
        concept_feature = self.load_feature_norms()
        feature_to_concepts = get_feature_to_concepts(concept_feature)

        features = sorted(feature_to_concepts.keys())
        feature_to_id = {feature: i for i, feature in enumerate(features)}

        features_selected = [
            norm for norm, concepts in feature_to_concepts.items() if len(concepts) >= 5
        ]
        # features_selected = features

        return feature_to_concepts, feature_to_id, features_selected

    def get_suffix(self):
        return "{}-{}".format(self.model, self.thresh)


class BinderDenseNormsLoader(NormsLoader):
    def __init__(self):
        self.model = "binder-dense"

    def load_concepts(self):
        path = "data/binder-norms.xlsx"
        path = DIR_LOCAL / path
        df = pd.read_excel(path)
        df = filter_by_things_concepts(df)
        return df["Word"].tolist()

    def __call__(self):
        df = load_binder_dense()
        data = df.to_dict("records")
        feature_to_concepts = {
            f: [(datum["Word"], datum["Value"]) for datum in group]
            for f, group in groupby(data, key=lambda x: x["Feature"])
        }

        features = sorted(feature_to_concepts.keys())
        feature_to_id = {feature: i for i, feature in enumerate(features)}
        features_selected = features

        return feature_to_concepts, feature_to_id, features_selected

    def get_suffix(self):
        return str(self.model)


NORMS_LOADERS = {
    "generated-gpt35": partial(GPT3NormsLoader, norms_model="chatgpt-gpt3.5-turbo"),
    "mcrae": McRaeNormsLoader,
    "mcrae-mapped": McRaeMappedNormsLoader,
    "mcrae-x-things": McRaeXThingsNormsLoader,
    "binder-3": partial(BinderNormsLoader, thresh=3),
    "binder-4": partial(BinderNormsLoader, thresh=4),
    "binder-5": partial(BinderNormsLoader, thresh=5),
    "binder-median": partial(BinderNormsLoader, thresh="median"),
    "binder-dense": BinderDenseNormsLoader,
}


CLASSIFIERS = {
    "linear-probe": partial(
        LogisticRegression,
        penalty=None,
        max_iter=1_000,
    ),
    "linear-regression": LinearRegression,
    "linear-probe-std": lambda: make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1_000),
    ),
    "knn-3": lambda: make_pipeline(
        Normalizer(norm="l2"),
        KNeighborsClassifier(n_neighbors=3),
    ),
    "mlp": lambda: make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(1024, 512)),
    ),
}

import networkx as nx 
import matplotlib.pyplot as plt
import community as community_louvain

@click.command()
@click.option(
    "-c",
    "--classifier-type",
    "classifier_type",
    type=click.Choice(CLASSIFIERS),
    required=True,
)
@click.option(
    "--embeddings-level",
    "embeddings_level",
    type=click.Choice(AGGREGATE_EMBEDDINGS),
    required=True,
)
@click.option(
    "-f",
    "--feature-type",
    "feature_type",
    type=click.Choice(FEATURE_TYPE_TO_MODALITY),
    required=True,
)
@click.option(
    "--norms-type",
    "norms_type",
    type=click.Choice(NORMS_LOADERS),
    required=True,
)
@click.option(
    "--split-type",
    "split_type",
    type=click.Choice(GET_TRAIN_TEST_SPLIT),
    required=True,
)
def main(classifier_type, embeddings_level, feature_type, norms_type, split_type):
    assert implies(classifier_type == "linear-regression", split_type != "repeated-k-fold")
    assert implies(classifier_type == "linear-regression", norms_type == "binder-dense")
    assert implies(norms_type == "binder-dense", classifier_type == "linear-regression")

    dataset_name = "things"
    dataset = DATASETS[dataset_name]()

    norm_loader = NORMS_LOADERS[norms_type]()
    feature_to_concepts, feature_to_id, features_selected = norm_loader()
    selected_concepts = norm_loader.load_concepts()
    selected_labels = [dataset.class_to_label[c] for c in selected_concepts]

    embeddings, labels = load_embeddings(dataset_name, feature_type, embeddings_level)
    matr = np.array(embeddings)
    x = matr.T
    res = (matr@x)
    np.fill_diagonal(res,-np.inf)
    values_similarity = []
    pair_value_max = []
    for i in range(len(res)):
        m= np.max(res[i])
        pair_value_max.append(np.argmax(res[i]))
        values_similarity.append(m)
    cop= [ (i,j) for i,j in enumerate(values_similarity)]
    values_similarity.sort()
    threshold = 600
    
    threshold_value = values_similarity[-threshold]
    print('threshold (>)',threshold_value)
    cop = list(filter( lambda x : x[1]> threshold_value,cop))

    
    edges = [ (c[0],pair_value_max[c[0]]) for c in cop]
    nodes = set((e[0] for e in edges) )
    nodes |= set((e[1] for e in edges))
    
    # get the graph
    decoder =open('./data/concepts-things.txt').read().split()
    nodes = map(lambda x : decoder[x],nodes)
    edges = map(lambda  x: (decoder[x[0]],decoder[x[1]]),edges)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)



            # Find communities
    partition = community_louvain.best_partition(G)

    # -------- compact community layout: no ring, filled center, tight clusters --------
    import math

    def community_layout_compact(
        G,
        partition,
        *,
        cluster_scale=8.0,     # global scale for cluster centers (smaller -> less ring)
        cluster_spread=0.45,   # radius of each cluster's internal layout (smaller -> tighter)
        gravity_w=0.7,         # pull cluster centers slightly toward the middle (0.3–1.0 works well)
        inter_boost=1.0,       # multiply inter-cluster edge weights (>=1 keeps clusters apart)
        seed=42,
    ):
        rng = np.random.default_rng(seed)

        # group nodes by community id
        clusters = {}
        for n, cl in partition.items():
            clusters.setdefault(cl, []).append(n)

        # meta-graph of clusters with edges weighted by count of inter-cluster edges
        H = nx.Graph()
        for cl in clusters:
            H.add_node(cl, size=len(clusters[cl]))

        for u, v in G.edges():
            cu, cv = partition[u], partition[v]
            if cu != cv:
                if H.has_edge(cu, cv):
                    H[cu][cv]["weight"] += 1
                else:
                    H.add_edge(cu, cv, weight=1)

        # (A) Add a light "gravity hub" to avoid the big empty center
        HUB = "__hub__"
        H.add_node(HUB)
        for cl in clusters:
            # connect hub to every cluster with a weak edge
            H.add_edge(HUB, cl, weight=gravity_w)

        # (B) Optionally boost inter-cluster edges to preserve separation
        if inter_boost != 1.0:
            for u, v, d in H.edges(data=True):
                if HUB not in (u, v):
                    d["weight"] *= inter_boost

        # (C) Layout cluster centers (including the hub), then drop the hub
        # A smaller scale + hub prevents the "ring" effect.
        k_global = None  # let networkx pick based on graph size
        pos_clusters_full = nx.spring_layout(
            H,
            seed=seed,
            scale=cluster_scale,
            weight="weight",
            k=k_global,
            center=(0.0, 0.0),
        )
        pos_clusters = {cl: pos for cl, pos in pos_clusters_full.items() if cl != HUB}

        # (D) Layout nodes inside each cluster tightly, then translate to cluster center
        pos = {}
        for cl, nodes in clusters.items():
            sub = G.subgraph(nodes)
            # keep clusters tight: small k & scale
            k_local = max(0.04, cluster_spread / math.sqrt(max(1, sub.number_of_nodes())))
            local = nx.spring_layout(
                sub,
                seed=int(rng.integers(0, 1_000_000)),
                k=k_local,
                scale=cluster_spread,
                center=(0.0, 0.0),
            )
            cx, cy = pos_clusters[cl]
            # tiny jitter to avoid perfect overlap of different clusters
            jx, jy = rng.normal(0, cluster_spread * 0.02), rng.normal(0, cluster_spread * 0.02)
            for n, (x, y) in local.items():
                pos[n] = (x + cx + jx, y + cy + jy)
        return pos

    # choose layout (tweakable knobs below)
    pos = community_layout_compact(
        G,
        partition,
        cluster_scale=7.5,    # ↓ to reduce outer spread & avoid ring
        cluster_spread=0.40,  # ↓ for tighter intra-cluster packing
        gravity_w=0.8,        # ↑ pulls clusters slightly inward (fills center)
        inter_boost=1.0,      # ≥1 keeps clusters apart; try 1.2 if clusters collapse
        seed=42,
    )

    # Color map per cluster
    clusters = sorted(set(partition.values()))
    cmap = plt.cm.tab20
    cluster_color_map = {cl: cmap(i % cmap.N) for i, cl in enumerate(clusters)}

    # Separate intra vs inter edges (for styling)
    intra_edges, inter_edges = [], []
    for u, v in G.edges():
        (intra_edges if partition[u] == partition[v] else inter_edges).append((u, v))

    plt.figure(figsize=(10, 10))

    # Nodes colored by cluster
    node_colors = [cluster_color_map[partition[n]] for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=60)

    # Intra-cluster edges: slightly thicker
    nx.draw_networkx_edges(G, pos, edgelist=intra_edges, alpha=0.55, width=.2)

    # Inter-cluster edges: thin & faint to visually de-emphasize
    nx.draw_networkx_edges(G, pos, edgelist=inter_edges, alpha=0.15, width=0.1, style="dotted")

    nx.draw_networkx_labels(G, pos, font_size=6)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"graph th {threshold}.png", dpi=300)


if __name__ == "__main__":
    main()
