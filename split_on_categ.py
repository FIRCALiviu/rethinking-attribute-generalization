
import json 
import pdb
from probing_norms.utils import *
from functools import partial
from probing_norms.data import DATASETS,load_mcrae_x_things,get_feature_to_concepts,DIR_LOCAL
import click
class NormsLoader:
    def __call__(self):
        raise NotImplementedError

    def get_suffix(self) -> str:
        raise NotImplementedError

    def load_concepts(self):
        path = "data/concepts-things.txt"
        path = DIR_LOCAL / path
        return read_file(path)

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

NORMS_LOADERS = {

    "mcrae-x-things": McRaeXThingsNormsLoader,

}
seed = 15

def get_binary_labels(labels, feature, feature_to_concepts, class_to_label):
    concepts_str = feature_to_concepts[feature]
    concepts_num = [class_to_label[c] for c in concepts_str]
    concepts_num = set(concepts_num)
    binary_labels = [label in concepts_num for label in labels]
    return np.array(binary_labels).astype(int)


import csv 
import pdb
file = open('../Categories_final_20200131.tsv')

rd = csv.reader(file,delimiter='\t',quotechar='"')

readall = [i for i in rd]
readall = readall[1:-1]


categs_freq = [0 ]*53

for i in readall:

    for j,exists in enumerate(i[2:]):

        if exists:
            categs_freq[j]+=1
        
assign = []
special_categ = 53 # for those with no category

def get_biggest_categ_index(row):
    index_max = -1
    max_categ = -1
    for i,categ in enumerate(row):
        if categ and categs_freq[i]>max_categ:
            index_max = i
            max_categ = categs_freq[i]
    if index_max == -1:
        return special_categ
    return index_max

for row in readall:
    categs = row[2:]
    idx = get_biggest_categ_index(categs)
    assign.append(idx)

categs_recount = {i:0 for i in range(0,special_categ+1)}
for i in assign:
    categs_recount[i]+=1
# pentru viz mai usoara
categs_recount_nozero  = {i:j for i,j in categs_recount.items() if j !=0}

import numpy as np
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


def verify(train_split,test_split):
    categs_in_test = set()
    categs_in_train = set()
    for i in train_split:
        categs_in_train.add(assign[i])
    for i in test_split:
        categs_in_test.add(assign[i])
    if len(categs_in_test&categs_in_train) != 0:
        print('[CRITICAL ERROR] verif failed intersect : ',categs_in_test&categs_in_train)
@click.command()
@click.option(
    "--norms-type",
    "norms_type",
    type=click.Choice(NORMS_LOADERS),
    required=True,
)
def main( norms_type):
    print('a')

    dataset_name = "things"
    dataset = DATASETS[dataset_name]()
    norm_loader = NORMS_LOADERS[norms_type]()
    feature_to_concepts, feature_to_id, features_selected = norm_loader()
    selected_concepts = norm_loader.load_concepts()
    
    selected_labels = [dataset.class_to_label[c] for c in selected_concepts]



    file = open('train_test_split.json','w')
    file2 = open('temp.json','w')

    all_splits = []
    all_features = []
    c = 0
    for feature in features_selected:
        binary_labels = get_binary_labels(np.array(range(len(assign))),feature,feature_to_concepts,dataset.class_to_label)
        n_tries = 10 # try to split 4 times
        success = False 
        while n_tries :
            n_tries -=1
            try:
                split = group_wise_train_test_split(assign,binary_labels,random_state=n_tries+50)
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
    json.dump({'splits':all_splits,'features':all_features},file)
    json.dump(all_features , file2)
    print('features not included:')
    print(list(set(features_selected)-set(all_features)))
main()



