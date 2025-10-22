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
import ast
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, RepeatedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.cluster import KMeans as k_means
from tqdm import tqdm
import time 
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
    idxss = train_test_split(idxs, test_size=0.2, random_state=seed)
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
            random_state=seed,
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
            random_state=seed,
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

    random.seed(seed)
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
import numpy as np
seed=int(time.perf_counter())
np.random.seed(seed)
def group_wise_train_test_split(assign, label,
                                train_frac=0.8,
                                max_rel_change=0.62,
                                random_state=seed):
    """
    Splits grouped data into train/test without breaking groups.

    Parameters:
    - assign: array-like of group IDs for each point (length N)
    - label:  array-like of 0/1 labels for each point (length N)
    - train_frac: fraction of total points desired in train (e.g. 0.8)
    - max_rel_change: maximum allowed relative change in positive rate
                      between train and overall (e.g. 0.2 = ±20%)
    - random_state: seed for reproducibility

    Returns:
    - 1D numpy array of 0/1 flags (1=train, 0=test)
    """
    rng = np.random.default_rng(random_state)
    assign = np.asarray(assign)
    label  = np.asarray(label)
    N_total = len(label)
    P_total = label.sum()
    global_pos_rate = P_total / N_total

    # Build per-group statistics
    stats = {}
    for g, y in zip(assign, label):
        stats.setdefault(g, {'n': 0, 'p': 0})
        stats[g]['n'] += 1
        stats[g]['p'] += y

    # Shuffle group order for randomness
    
    groups = list(stats.keys())
    rng.shuffle(groups)

    target_n = train_frac * N_total
    train_groups = set()
    n_train = 0
    p_train = 0

    # Greedy selection of groups
    for g in groups:
        ng = stats[g]['n']
        pg = stats[g]['p']
        # Skip if overshoots too far
        if n_train + ng > target_n and n_train < target_n:
            if n_train + ng - target_n > 0.05 * N_total:
                continue

        # Compute new positive-rate and relative change
        new_p = p_train + pg
        new_n = n_train + ng
        new_rate = new_p / new_n
        rel_change = abs(new_rate - global_pos_rate) / global_pos_rate

        # Accept group if within balance tolerance
        if rel_change <= max_rel_change:
            train_groups.add(g)
            n_train = new_n
            p_train = new_p

        # Stop once close to desired size
        if n_train >= target_n * 0.98:
            break

    # Build mask: 1 for train, 0 for test
    mask=  np.isin(assign, list(train_groups)).astype(int)
    N_train = mask.sum()
    train_frac_actual = N_train / N_total
    pos_train = label[mask==1].sum() / N_train
    rel_change = abs(pos_train - global_pos_rate) / global_pos_rate

    if abs(train_frac_actual - train_frac) > 0.5:
        raise ValueError(f"Split failed: train fraction {train_frac_actual:.3f} "
                         f"outside ±5% of target {train_frac:.2f}")
    if rel_change > max_rel_change:
        raise ValueError(f"Split failed: relative class‑balance change {rel_change:.3f} "
                         f"exceeds tolerance {max_rel_change:.3f}")
    return mask

def get_assign_k_means(embeddings,k):
	algorithm = k_means(n_clusters = k,random_state=seed).fit(embeddings)
	return algorithm.labels_
def verify(train_split,test_split):
    categs_in_test = set()
    categs_in_train = set()
    for i in train_split:
        categs_in_train.add(assign[i])
    for i in test_split:
        categs_in_test.add(assign[i])
    if len(categs_in_test&categs_in_train) != 0:
        print('[CRITICAL ERROR] verif failed interset : ',categs_in_test&categs_in_train)
assign = None
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
    global assign
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
    k= 100
    assign = get_assign_k_means(embeddings,k)
    all_splits = []
    all_features = []
    c = 0
    
    all_splits = []
    all_features = []
    c = 0
    for feature in features_selected:
        binary_labels = get_binary_labels(np.array(range(len(assign))),feature,feature_to_concepts,dataset.class_to_label)
        n_tries = 4 # try to split 4 times
        success = False 
        while n_tries :
            n_tries -=1
            try:
                split = group_wise_train_test_split(assign,binary_labels,random_state=seed+n_tries+50)
                train = np.array(range(len(split)))[split.astype(bool)]
                test =  np.array(range(len(split)))[np.logical_not(split)]
                if len(train)>len(test):
                    
                    n_tries = 0
                    success = True
            except ValueError:
                pass
            
        
        if success:
            verify(train,test)
            all_splits.append((train.tolist(),test.tolist()))
            all_features.append(feature)
        else:
            c+=1
    print("Number of ignored datasets : ", c)
    constraint = ast.literal_eval(open('./intersect_features.json').read())
    if len( set(constraint) - set(all_features)) >7 :
        raise RuntimeError(f"prea putine,{len( set(constraint) - set(all_features))}")
        exit(-1)
    file = open(f'./k_means_split.json','w')
    file2 = open(f'./k_means_features.json','w')
    json.dump({'splits':all_splits,'features':all_features},file)
    json.dump(all_features , file2)
    print('features not included:')
    print(list(set(features_selected)-set(all_features)))
if __name__ == "__main__":
    main()
