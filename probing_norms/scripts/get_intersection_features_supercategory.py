
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



import numpy as np
np.random.seed(seed)



def get_supercategs():
    file= open('/root/Categories_final_20200131.tsv','r')
    rd = csv.reader(file,delimiter='\t',quotechar='"')

    readall = [i for i in rd]
    supercategs = readall[0][2:]

    supercategs_to_concepts = defaultdict(set)
    
    
    readall = readall[1:-1]
    for i in readall:

        for j,val in enumerate(i[2:]):

            if val:
                supercategs_to_concepts[supercategs[j]].add(i[0].replace(" ",'_'))
    return supercategs_to_concepts,supercategs
        

def main( ):
    norms_type = 'mcrae-x-things'


    dataset_name = "things"
    dataset = DATASETS[dataset_name]()
    norm_loader = NORMS_LOADERS[norms_type]()
    feature_to_concepts, feature_to_id, features_selected = norm_loader()
    selected_concepts = norm_loader.load_concepts()
    supercategs_to_concepts,supercategs = get_supercategs()
    concepts = defaultdict(list)
    for category in supercategs:
        for concept in supercategs_to_concepts[category]:
            concepts[concept].append(category)
    # has_4 = [(k,v) for k,v in concepts.items() if len(v)>=4]
    # has_1 = [(k,v) for k,v in concepts.items() if len(v)==1]
    # x = has_4[::1][:100]
    # for i in x:
    #     print(i[0],*i[1],sep=',')
    
    # x = has_1[::1][:100]
    # for i in x:
    #     print(i[0],":",*i[1])
if __name__=='__main__':
    main()



