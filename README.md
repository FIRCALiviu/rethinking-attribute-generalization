This repository contains code for the paper:
>   Liviu Nicolae Firca, Antonio Barbalau, Dan Oneata, Elena Burceanu  
>[**Not All Splits Are Equal: Rethinking Attribute
Generalization Across Dissimilar Categories**](https://arxiv.org/abs/2509.06998)
> CausScien & Reliable ML from Unreliable Data, 2025

## Abstract

Can models generalize attribute knowledge across semantically and perceptually dissimilar categories? While prior work has addressed attribute prediction within narrow taxonomic or visually similar domains, it remains unclear whether current models can abstract attributes and apply them to conceptually distant categories. This work presents the first explicit evaluation for the robustness of the attribute prediction task under such conditions, testing whether models can correctly infer shared attributes between unrelated object types: e.g., identifying that the attribute "has four legs" is common to both "dogs" and "chairs". To enable this evaluation, we introduce train-test split strategies that progressively reduce correlation between training and test sets, based on: LLM-driven semantic grouping, embedding similarity thresholding, embedding-based clustering, and supercategory-based partitioning using ground-truth labels. Results show a sharp drop in performance as the correlation between training and test categories decreases, indicating strong sensitivity to split design. Among the evaluated methods, clustering yields the most effective trade-off, reducing hidden correlations while preserving learnability. These findings offer new insights into the limitations of current representations and inform future benchmark construction for attribute reasoning.

## Splits

The splits have been cached in the `splits` directory.



# Reproducing the results

### Setup

Install dependencies via conda:

```bash
conda create -n probing-norms python=3.12
conda activate probing-norms
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

Then you need to install RAPIDS 25.1 in the environment. 
Then you can install this code as a library with:

```bash
pip install -e .
```


**Data.**
You will need to download the [THINGS dataset](https://osf.io/jum2f/files/osfstorage), for example using `wget`:
```bash
wget https://files.osf.io/v1/resources/jum2f/providers/osfstorage/?zip= -O things.zip
```

## Evaluate the performance of a model on a split


To evaluate the performance of a single model on a splitting stategy (let's say the Swin-V2 model, code as `swin-v2-ssl` on the super), we have to perform three steps.

**Feature extraction.**
Extract the features for the THINGS concepts:
```bash
python probing_norms/extract_features_image.py -d things -f swin-v2-ssl
```

**Model training.**
The next step is to train linear probes for both the McRae × THINGS and the Binder datasets:
```bash
python probing_norms/predict.py --feature-type swin-v2-ssl --norms-type mcrae-x-things --split-type repeated-k-fold --embeddings-level concept --classifier-type linear-probe
```
where `--split-type` can be `repeated-k-fold` for random splits, `similarity-k-fold` for obtaining the similarity split, `llm-k-fold` for obtaining the LLM based one, `clustering` for the clustering split and `supercategs` for the supercategory split.

**Results generation.**
Finally, we evaluate the performance in terms of F₁ selectivity for McRae × THINGS
```bash
python probing_norms/get_results.py paper-table-main-acl-camera-ready:swin-v2-ssl
```

**Correlation with the supercategory**

```bash
python probing_norms/scripts/eval_norm_correlations.py
```

