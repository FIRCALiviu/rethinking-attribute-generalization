import csv
import json
import ast
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import pandas as pd
import pdb
from probing_norms.predict import McRaeMappedNormsLoader
from probing_norms.get_results import (
    FEATURE_NAMES,
    MAIN_TABLE_MODELS,
    get_score_random_features,
    load_result_features,
    load_taxonomy_mcrae,
)

# from https://github.com/ViCCo-Group/THINGS-data/blob/main/MRI/data/Categories_final_20200131.tsv
THINGS_SUPERCATEGORIES = "data/things/Categories_final_20200131.tsv"

# first row in the file is (here for reference):
SUPER_CATS = [
    "animal" "bird",
    "body part",
    "clothing",
    "clothing accessory",
    "container",
    "dessert",
    "drink",
    "electronic device",
    "food",
    "fruit",
    "furniture",
    "home decor",
    "insect",
    "kitchen appliance",
    "kitchen tool",
    "medical equipment",
    "musical instrument",
    "office supply",
    "part of car",
    "plant",
    "sports equipment",
    "tool",
    "toy",
    "vegetable",
    "vehicle",
    "weapon",
    "candy",
    "hardware",
    "mammal",
    "sea animal",
    "fastener",
    "jewelry",
    "watercraft",
    "condiment",
    "garden tool",
    "lighting",
    "personal hygiene item",
    "scientific equipment",
    "arts and crafts supply",
    "home appliance",
    "seafood",
    "breakfast food",
    "footwear",
    "safety equipment",
    "school supply",
    "headwear",
    "protective clothing",
    "construction equipment",
    "game",
    "farm animal",
    "outerwear",
    "women's clothing",
]


def get_supercategories(file=THINGS_SUPERCATEGORIES):
    """Reads the supercategories file and returns a dictionary of supercategories to concepts.

    The file is formatted: word | definition | supercategories... [with 1 for assigned supercategory]
    Not all concepts have supercategories (1): 407 are missing, we ignore these words.

    """
    supercategories = defaultdict(set)
    missing = 0
    with open(file, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        supercat_set = header[2:]
        for row in reader:
            word = row[0]
            try:
                supercat_id = (row[2:]).index("1")
            except ValueError:
                print(f"Error: {row[0]} {''.join(row[2:])}")
                missing += 1
                continue
            supercategories[supercat_set[supercat_id]].add(word)

        # Future: there are a few very small supercategories (headwear, garden tool) that might need to be removed
        # they cause problems for overlap correlation more than jaccard and intersection.
        num_words = sum([len(supercategories[x]) for x in supercategories])
        print(f"Found supercategories for {num_words} words")
        print(f"Missing supercategories for {missing} words")
        print(f"Found {len(supercategories)} supercategories")
        # print("supercategories: ", supercategories.keys())
        # for supercat in supercategories:
        #     print(f"supercategory: {supercat} {len(supercategories[supercat])} words")

    return supercategories


def load_result_features_1(model):
    classifier_type = "linear-probe"
    embeddings_level = "concept"
    split_type = "repeated-k-fold"
    norms_type = "mcrae-x-things"
    results = load_result_features(
        classifier_type,
        embeddings_level,
        split_type,
        model,
        norms_type,
    )
    results_random = get_score_random_features(norms_type)
    for result in results:
        result["score"] = result["score-f1"] - results_random[result["feature"]]
    return results


def get_best_supercategory_matches():
    """Matches each feature to a supercategory based on the feature's category set.
    We use the supercategory with the highest intersection with the feature's category set,
    which is later normalized by the size of the category set.
    """

    supercategories = get_supercategories()
    norms_loader = McRaeMappedNormsLoader()
    feature_to_concepts, _, features_selected = norms_loader()
    
    
    
    #features_selected = ast.literal_eval(open('./k_means_features.json').read())
    feature_to_concepts = {
        f: set(cs) for f, cs in feature_to_concepts.items() if f in features_selected
    }

    supercat_names = list(supercategories.keys())

    """Possible ways of evaluating best set overlap: 
        - intersection / feature extension (concept set) size: best_intersection
        - intersection / union: best_jacard
        - intersection / min: best_overlap_coeff
    Jacard and especially overlap coefficient get distracted by small supersets, which is not what we want:
    the question here is which supercategory is the biggest overlap for a given feature.
    """

    best_match_supercat = {}
    for feature in feature_to_concepts:
        feature_size = len(feature_to_concepts[feature])
        intersections = []
        # jacards = []  # intersection over union
        # overlap_coeffs = []  # intersection over min
        for supercat in supercategories:
            supercat_concepts = supercategories[supercat]
            feature_concepts = feature_to_concepts[feature]
            intersection = len(feature_concepts & supercat_concepts)
            intersections.append(intersection)
            # jacard = intersection / len(feature_concepts | supercat_concepts)
            # overlap_coeff = len(common_concepts) / min(len(feature_concepts), len(supercat_concepts))
            # jacards.append(jacard)
            # overlap_coeffs.append(overlap_coeff)
        best_intersection = max(intersections)
        best_intersection_idx = intersections.index(best_intersection)
        best_intersection_cat = supercat_names[best_intersection_idx]
        best_intersection_norm = best_intersection / feature_size
        # best_jacard = max(jacards)
        # best_jacard_idx = jacards.index(best_jacard)
        # best_jacard_cat = supercat_names[best_jacard_idx]
        # best_overlap_coeff = max(overlap_coeffs)
        # best_overlap_coeff_idx = overlap_coeffs.index(best_overlap_coeff)
        # best_overlap_coeff_cat = supercat_names[best_overlap_coeff_idx]

        # if best_jacard_cat != best_intersection_cat:
        #     print(f"Feature: {feature} best_intersection: {best_intersection_norm} {best_intersection_cat} best_jacard: {best_jacard} {best_jacard_cat} best_overlap_coeff: {best_overlap_coeff} {best_overlap_coeff_cat}")
        #     print(f"         concepts {feature_to_concepts[feature]}")

        best_match_supercat[feature] = (best_intersection_cat, best_intersection_norm)

    return best_match_supercat

def beautify(taxonomy):
    match taxonomy:
        case "visual-colour":
            return "visual : colour"
        case "visual-form_and_surface":
            return "visual : form & surface"
        case "visual-motion":
            return "visual : motion"
        case _:
            return taxonomy

def correct_placement(feature):
    match feature:
        case "made_of_metal":
            return "top"
        case "has_feaders":
            return "top"
        case "a_rodent":
            return "top"
        case _ :
            return "bottom"

def main():
    best_match_supercat = get_best_supercategory_matches()
    bms_records = [(k, v[0], v[1]) for k, v in best_match_supercat.items()]
    df_features = pd.DataFrame.from_records(
        bms_records,
        columns=["feature", "supercategory", "intersection_norm"],
    )

    taxonomoy = load_taxonomy_mcrae()
    df_features["taxonomy"] = df_features["feature"].map(taxonomoy)
    # print(df_features.groupby("taxonomy")["intersection_norm"].agg(["mean", "size"]))

    def get_correlation_model(model):
        results = load_result_features_1(model)
        df_scores = pd.DataFrame(results)
        df_scores = df_scores[["feature", "score"]]
        df = pd.merge(df_features, df_scores, on="feature")

        # Pearson correlation between intersection_norm and score
        
        corr = df[["intersection_norm", "score"]].corr()
        k= 30
        X = (df['intersection_norm'] * 100).to_numpy()
        Y = df['score'].to_numpy()

        
        import matplotlib
        matplotlib.rcParams['xtick.labelsize'] = 30
        matplotlib.rcParams['ytick.labelsize'] = 30

        fig, ax = plt.subplots(figsize=(30, 20))

        # scatter by taxonomy (colors + legend auto)
        for tax, sub in df.groupby('taxonomy'):
            ax.scatter(sub['intersection_norm'] * 100, sub['score'], s=200, label=beautify(str(tax)))

        # ==== Best-fit line + correlation ====
        x = (df['intersection_norm'] * 100).to_numpy()
        y = df['score'].to_numpy()

        # linear fit (least squares)
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 200)
        y_line = slope * x_line + intercept


        
       

        
        for i, rrow in df.iterrows():
            if  i%17 == 0: 
                ax.annotate(
                    str(rrow['feature']),
                    xy=(rrow['intersection_norm'] * 100, rrow['score']),      # point
                    xytext=(10, 10),                                          # offset in points
                    textcoords='offset points',
                    fontsize=30,
                    ha='left', va=correct_placement(str(rrow['feature'])),
                    arrowprops=dict(arrowstyle='-', color='black', lw=0.8)
                )

        # --- updated labels & title ---
        plt.title("Embedding Clustering", fontweight='bold', fontsize=60)
        plt.xlabel('Supercategory inclusion (%)', fontsize=40)
        plt.ylabel('F1 selectivity', fontsize=40)

        ax.legend(title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=38, title_fontsize=50)
        ax.grid()
        plt.tight_layout()
        plt.savefig('correlation_plot.png', dpi=50)

        
        return corr.loc['score', 'intersection_norm']

    results = [
        {
            "model": model,
            "correlation": get_correlation_model(model),
        }
            for model in ["swin-v2-ssl"]
    ]
    df = pd.DataFrame.from_records(results)
    df["model"] = df["model"].map(FEATURE_NAMES)
    df = df.sort_values(by="correlation", ascending=False)
    print(df.to_string(index=False))
    print(df.to_latex(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
