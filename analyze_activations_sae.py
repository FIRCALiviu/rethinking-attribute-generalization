import torch 
import pdb 
from pathlib import Path
from probing_norms.utils import read_file, reverse_dict, cache_json
from probing_norms.data import load_mcrae_x_things,get_feature_to_concepts
from sklearn.metrics import f1_score 
from multiprocessing import Pool
from cuml import PCA

def all_instances_idxs(iterable,values):
    values = set(values)
    return [i for i,v in enumerate(iterable) if v in values]

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


def prepare_metadata(root):
        def get_class_name(path):
            _, class_name, _ = path.split("/")
            return class_name

        metadata_path = str(root / "01_image-level" / "image-paths.csv")
        image_files = read_file(metadata_path)

        classes = [get_class_name(path) for path in image_files]
        label_to_class = dict(enumerate(sorted(set(classes))))
        class_to_label = reverse_dict(label_to_class)

        labels = [class_to_label[c] for c in classes]
        return label_to_class,class_to_label



activations = torch.load('./data/activations_sae.pt')
activations = PCA().fit_transform(activations.cpu().numpy())

image_label = open('./data/labels_images.txt').read().split()

label_to_concept,_ = prepare_metadata(Path("/root/data/things"))
images_concepts = list(map( lambda x : label_to_concept[int(x)] , image_label))

feature_to_concepts, feature_to_id, features_selected = McRaeXThingsNormsLoader() ()


def analyze_feature_activation(feature_studied):
    work = []
    work.append('\n'+'='*9+ f'\nFor feature {feature_studied}\n')
    has_feature = feature_to_concepts[feature_studied]
    idxs = all_instances_idxs(images_concepts,has_feature)

    y_true = torch.tensor([False]*len(image_label))
    y_true[idxs] = True 
    for neuron in range(activations.shape[1]): # each neuron is an activation in SAE, ie a 'feature' detected
        activation_neuron = activations[:,neuron]
        y_pred = activation_neuron > 0
        score = f1_score(y_true,y_pred)
        if score > 0.4:
            work.append(f"feature : {feature_studied} neuron {neuron} score {score}\n")
    if len(work) == 1 :
        work.append('\n__XX_NONE__')
    return "".join(work)
features = eval(open('intersect_features.json').read())

with Pool(38) as p:
        for result in p.imap_unordered(analyze_feature_activation, features):
            print(result, flush=True)
        