import time
import json
import pdb
import random
import os
import pickle
import ast
from collections import defaultdict
from dataclasses import dataclass
from itertools import groupby
from functools import partial
from typing import Dict, List

import click
import numpy as np
import pandas as pd
import torch 

from sklearn.linear_model import LinearRegression
from cuml.linear_model import LogisticRegression
import cupy as cp
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, RepeatedKFold, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
#from sklearn.linear_model import LogisticRegression
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




class PCA:
    def __init__(self, n_components=None, variance_ratio=None, device=None):
        """
        Args:
            n_components (int): number of principal components to keep.
            variance_ratio (float): minimum fraction of variance to preserve (e.g., 0.96).
            device (str or torch.device): "cuda" or "cpu". Defaults to GPU if available.
        """
        self.n_components = n_components
        self.variance_ratio = variance_ratio
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.k_ = None
        self.device = device or 'cpu' 

    def fit(self, X):
        # Move data to GPU if available
        X = X.to(self.device)

        # Center the data
        self.mean_ = X.mean(dim=0)
        X_centered = X - self.mean_

        # SVD decomposition (torch.linalg.svd is more stable than torch.svd)
        U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
        V = Vh.T

        # Explained variance
        explained_variance = (S**2) / (X_centered.shape[0] - 1)
        total_variance = explained_variance.sum()
        explained_variance_ratio = explained_variance / total_variance

        # Determine number of components
        if self.n_components is not None:
            k = self.n_components
        elif self.variance_ratio is not None:
            cumulative = torch.cumsum(explained_variance_ratio, dim=0)
            k = int((cumulative < self.variance_ratio).sum().item() + 1)
        else:
            k = X.shape[1]  # keep all

        # Store results
        self.k_ = k
        self.components_ = V[:, :k]
        self.explained_variance_ = explained_variance[:k]
        self.explained_variance_ratio_ = explained_variance_ratio[:k]

        return self

    def transform(self, X):
        X = X.to(self.device)
        X_centered = X - self.mean_
        return torch.mm(X_centered, self.components_)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def get_idxs_similar():
    with open('./temp.txt') as f:
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


def load_embeddings():
    activations = torch.load('./data/activations_sae.pt').detach().numpy()
    image_label = np.array([ int(x) for x in  open('./data/labels_images.txt').read().split()])
    unique_labels = np.sort(np.unique(image_label))
    mean_embeddings = np.empty((unique_labels.shape[0],activations.shape[-1]))
    for i in unique_labels :
        idxs = image_label ==i 
        mean_embeddings[i] = activations[idxs].mean(axis=0)


    return mean_embeddings,unique_labels

def predict_func_classifier(clf, X_te):
    return clf.predict_proba(X_te)[:, 1]


def predict_func_regression(clf, X_te):
    return clf.predict(X_te)


PREDICT_FUNCS = defaultdict(lambda: predict_func_classifier)
PREDICT_FUNCS["LinearRegression"] = predict_func_regression
feature_num_XX = 1
log_file = open('log_stats.txt','w')
with open('./temp_neurons','rb') as f:
    feature_neurons_allowed = pickle.load(f)

def predict1(make_classifier, X, y, split,log_statistics=False):
    global feature_num_XX
    idxs_tr = split.tr_idxs
    
    idxs_te = split.te_idxs

    feature = split.metadata['feature']
    neurons = feature_neurons_allowed[feature]
    X_filtered = X[:,neurons] 
    X_tr, y_tr = X_filtered[idxs_tr], y[idxs_tr]
    X_te, y_te = X_filtered[idxs_te], y[idxs_te]
    if log_statistics:
        tr_ratio = sum(y_tr)/(len(y_tr))
        te_ratio = sum(y_te) / len(y_te)

        log_file.write(f"  {len(y_tr)/(len(y_te)+len(y_tr))}  { (te_ratio-tr_ratio)/tr_ratio*100} \n")
        log_file.flush()
    feature_num_XX +=1
    clf = make_classifier()
    predict_func = PREDICT_FUNCS[clf.__class__.__name__]

    clf.fit(X_tr, y_tr)
    y_pr = predict_func(clf,X_te)
    
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
            **predict1(make_classifier, X, y, split,log_statistics=True),
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
    return Split(np.array(list(s_train)),np.array(list(s_test)),split.metadata)






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
    n_repeats=1,
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
            random_state=110,
        )

        return [
            Split(
                *idxss,
                metadata={"feature": feature},
            )
            for idxss in rskf.split(binary_labels, binary_labels)
        ]

    return {feature: get_f(feature) for feature in features}


def get_train_test_split_random(
    labels,
    features,
    *,
    feature_to_concepts,
    class_to_label,
    n_splits=1,
    n_repeats=1):
    def get_f(feature):
        binary_labels = get_binary_labels(
            labels,
            feature,
            feature_to_concepts,
            class_to_label,
        )
        nr = 10
        rskf = StratifiedShuffleSplit(
            n_splits=1,
            random_state=nr,
            test_size= 0.2
        )

        return [Split(*idxs,metadata = {"feature":feature}) for idxs in rskf.split(binary_labels, binary_labels)]

    return {feature : get_f(feature) for feature in features}

def get_train_test_split_k_means(
    labels,
    features,
    *,
    feature_to_concepts,
    class_to_label,
    n_splits=5,
    n_repeats=2):
    file = open('./k_means_split.json')
    loaded_data = json.load(file)
    splits = loaded_data['splits']
    features = loaded_data['features']
    def copy_splits(split
    ):
        return [Split(tr_idxs=np.array(split[0],dtype = np.int32),te_idxs=np.array(split[1],dtype=np.int32),metadata={}) ]

    return {feature : copy_splits(split) for feature,split in zip(features,splits)}

def get_train_test_split_categs(
    labels,
    features,
    *,
    feature_to_concepts,
    class_to_label,
    n_splits=5,
    n_repeats=2):
    file = open('./train_test_split.json')
    loaded_data = json.load(file)
    splits = loaded_data['splits']
    features = loaded_data['features']
    def copy_splits(split
    ):
        return [Split(tr_idxs=np.array(split[0],dtype = np.int32),te_idxs=np.array(split[1],dtype=np.int32),metadata={}) ]

    return {feature : copy_splits(split) for feature,split in zip(features,splits)}

def get_train_test_split_k_fold_excluded(
    labels,
    features,
    *,
    feature_to_concepts,
    class_to_label,
    n_splits=5,
    n_repeats=1,
):

    def get_f(feature):
        binary_labels = get_binary_labels(
            labels,
            feature,
            feature_to_concepts,
            class_to_label,
        )
        nr = 10
        rskf = StratifiedShuffleSplit(
            n_splits=n_splits,
            random_state=nr,
            test_size= 0.6
        )
        rskf2 = StratifiedShuffleSplit(
            n_splits=n_splits,
            random_state=nr,
            test_size=0.6
        )
        tz = [exclude_from_test(Split(
                *idxss,
                metadata={"feature": feature},
            ),get_idxs_similar())
            for idxss in rskf.split(binary_labels, binary_labels)
        ]
        return tz
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
            exclude_from_test(Split(
                *idxss,
                metadata={"feature": feature},
            ) ,get_idxs_similar())
            for idxss in rskf.split(indices)
        ]

    return {feature: get_f(feature) for feature in features}

GET_TRAIN_TEST_SPLIT = {
    "repeated-k-fold":  get_train_test_split_random,
    "repeated-k-fold-simple": get_train_test_split_k_fold_simple,
    "excluded-k-fold" : get_train_test_split_k_fold_excluded,
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
        max_iter=5_000,
        class_weight= 'balanced',
        penalty = 'l1',
        C = 0.1,
        tol = 10**-3

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
    features_selected = ast.literal_eval(open('./intersect_features.json').read())
    selected_concepts = norm_loader.load_concepts()
    selected_labels = [dataset.class_to_label[c] for c in selected_concepts]
    
    embeddings, labels = load_embeddings()
    idxs = np.isin(labels, selected_labels)

    embeddings = embeddings[idxs]
    labels = labels[idxs]

    splits = GET_TRAIN_TEST_SPLIT[split_type](
        labels,
        features_selected,
        feature_to_concepts=feature_to_concepts,
        class_to_label=dataset.class_to_label,
    )
    
    FOLDER = "/root/output/{}-predictions/{}/{}".format(
        classifier_type, embeddings_level, split_type
    )
    os.makedirs(FOLDER, exist_ok=True)

    def get_path(feature):
        feature_norm = norm_loader.get_suffix()
        feature_id = feature_to_id[feature]
        return "{}/{}-{}-{}-{}".format(
            FOLDER,
            dataset_name,
            feature_type,
            feature_norm,
            feature_id,
        )

    
    make_classifier = CLASSIFIERS[classifier_type]

    GET_LABELS_FUNCS = {
        "linear-regression": get_continuous_labels,
    }
    get_labels_func = GET_LABELS_FUNCS.get(classifier_type, get_binary_labels)

    def save_preds(path, results):
        data = [{k: r[k] for k in ("split", "preds")} for r in results]
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
 
    def save_clfs(path, results):
        data = [{k: r[k] for k in ("split", "clf")} for r in results]
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def cache_clf_and_preds(path, func, *args):
        path_json = path + ".json"
        results = func(*args)
        save_preds(path_json, results)
        # Avoid saving classifiers for KNN since it stores the whole data!
        # if classifier_type != "knn-3":
        #     save_clfs(path, results)
        
    def process_feature(feature):
        classifier_labels = get_labels_func(
            labels,
            feature,
            feature_to_concepts,
            dataset.class_to_label,
        )
        cache_clf_and_preds(
            get_path(feature),
            predict_splits,
            make_classifier,
            embeddings,
            classifier_labels,
            splits[feature],
        )

    # with Pool(16) as pool:
    #     processes = pool.imap(process_feature, features_selected)
    #     total = len(features_selected)
    #     for _ in tqdm(processes, total=total):
    #         pass

    for f in tqdm(features_selected):
        process_feature(f)


if __name__ == "__main__":
    main()
