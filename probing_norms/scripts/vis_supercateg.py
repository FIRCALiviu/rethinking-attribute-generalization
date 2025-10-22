from __future__ import annotations
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

def aggregate_by_labels(embeddings, labels):
    unique_labels = np.unique(labels)
    agg_embeddings = np.zeros((len(unique_labels), embeddings.shape[1]))
    for i, label in enumerate(unique_labels):
        idxs = labels == label
        agg_embeddings[i] = embeddings[idxs].mean(axis=0)
    return agg_embeddings, unique_labels
output = np.load('/root/output/features-image/things-swin-v2-ssl.npz', allow_pickle=True)
embeddings = output["X"]
labels = output["y"].astype(np.int32)
embeddings,labels= aggregate_by_labels(embeddings,labels)
names = open('./data/concepts-things.txt', 'r').read().split()

##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# ---- 1) Label-aware projection (UMAP->LDA->TSNE->PCA) ----
def project_2d(X, y, method="auto", random_state=42):
    y = np.asarray(y)
    method = method.lower()

    


    # LDA (label-aware, spreads classes; works even without UMAP)
    if method in ("auto","lda"):

            Y = LDA(n_components=2).fit_transform(X, y)
            return Y, "LDA"


    # t-SNE (cosine works well on embeddings)
    if method in ("auto","tsne"):
        n = len(X)
        perplexity = max(5, min(30, (n-1)//4))
        Y = TSNE(n_components=2, init="pca", learning_rate="auto",
                 perplexity=perplexity, metric="cosine",
                 random_state=random_state).fit_transform(X)
        return Y, "t-SNE"

    # PCA fallback
    Y = PCA(n_components=2, random_state=random_state).fit_transform(X)
    return Y, "PCA"


# ---- 2) One-vs-all small-multiples view ----

from matplotlib.transforms import Bbox
def focus_grid(
    Y, labels, names=names, *,
    active_labels=[12,5,21,22,28,25,3]

,                 # which clusters are colored
    labelable_labels=None,           # which clusters get name annotations (defaults to active)
    top=16, cols=4, point_size=200,
    bg_alpha=0.06, fg_alpha=0.95, jitter=0.0, random_state=42,
    cmap_name="tab20", savepath="1.png",
    show_centroids=False, show_legend=False,
    names_per_cluster=10, name_fontsize=7,
    strict=False,
    # --- new anti-overlap controls ---
    label_repel=True,
    label_repel_pixels=4,               # step size (pixels) per iteration
    label_repel_max_iter=200,           # max iterations
    label_pad_pixels=2,                 # pad around each text bbox
    draw_leader_lines=True,             # draw faint lines from point->label
):
    """
    Overlay plot of clusters. Exactly `names_per_cluster` labels per labelable cluster.
    If `label_repel=True`, text labels are iteratively nudged apart to reduce overlaps.
    """

    Y = np.asarray(Y); labels = np.asarray(labels)

    if names is not None:
        names = np.asarray(names)
        if len(names) != len(Y):
            raise ValueError("Length of 'names' must match number of rows in Y.")
    else:
        names = labels.astype(str)

    uniq, counts = np.unique(labels, return_counts=True)

    def _validate_pick(pick):
        if pick is None:
            return None
        pick = list(pick)
        present = set(uniq.tolist())
        missing = [lab for lab in pick if lab not in present]
        if missing and strict:
            raise ValueError(f"Labels not found in data: {missing}")
        return [lab for lab in pick if lab in present]

    # clusters to color
    if active_labels is not None:
        show = np.array(_validate_pick(active_labels), dtype=uniq.dtype)
        if show.size == 0:
            raise ValueError("No valid labels in 'active_labels'.")
    else:
        order = np.argsort(counts)[::-1][:top]
        show = uniq[order]

    # clusters that get name annotations (default = colored ones)
    labelable = _validate_pick(labelable_labels)
    labelable_set = set(show.tolist()) if labelable is None else set(labelable)

    rng = np.random.default_rng(random_state)
    if jitter > 0:
        Y = Y + rng.normal(scale=jitter, size=Y.shape)

    fig, ax = plt.subplots(figsize=(20, 20))

    # faint background
    ax.scatter(Y[:, 0], Y[:, 1], s=point_size, alpha=bg_alpha, c="white", linewidths=0)

    cmap = plt.get_cmap(cmap_name)
    handles, legend_labels = [], []

    # small initial text offset
    span = np.ptp(Y, axis=0)
    text_dx, text_dy = 0.006 * span[0], 0.006 * span[1]

    texts = []          # matplotlib.text.Text objects to repel
    anchors = []        # (x_point, y_point) anchor for leader lines
    lines = []          # line artists (optional)

    for i, lab in enumerate(show):
        mask = (labels == lab)
        idx = np.flatnonzero(mask)
        color = cmap(i % cmap.N)

        h = ax.scatter(
            Y[mask, 0], Y[mask, 1],
            s=point_size * 1.6, alpha=1, linewidths=0, c=[color]
        )
        handles.append(h)
        legend_labels.append(f"{lab} ({mask.sum()})")

        if show_centroids and mask.any():
            cx, cy = Y[mask].mean(axis=0)
            ax.text(cx, cy, str(lab), ha="center", va="center",
                    fontsize=8, weight="bold", color='black', zorder=3)

        # annotate exactly `names_per_cluster` points in allowed clusters
        if (lab in labelable_set) and idx.size:
            k = min(names_per_cluster, idx.size)
            picked_indices = rng.choice(idx, size=k, replace=False)
            for j in picked_indices:
                x, y = Y[j]
                t = ax.text(
                    x + text_dx, y + text_dy,
                    str(names[j]),
                    fontsize=name_fontsize, color='black',
                    ha="left", va="bottom", zorder=3
                )
                texts.append(t)
                anchors.append((x, y))
                if draw_leader_lines:
                    ln, = ax.plot([x, t.get_position()[0]], [y, t.get_position()[1]],
                                  linewidth=0.5, alpha=0.6, c='gray', zorder=2)
                    lines.append(ln)

    # ---------- repel labels to reduce overlap ----------
    if label_repel and texts:
        # ensure a renderer exists
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        def _pad_bbox(bb, pad):
            return Bbox.from_extents(bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad)

        for it in range(label_repel_max_iter):
            moved = False
            # current bboxes (with padding) & centers in pixel space
            bbs = [_pad_bbox(t.get_window_extent(renderer=renderer), label_pad_pixels) for t in texts]
            centers = [((bb.x0 + bb.x1) * 0.5, (bb.y0 + bb.y1) * 0.5) for bb in bbs]

            disp = np.zeros((len(texts), 2), dtype=float)  # pixel displacements to apply this iter

            # pairwise repulsion for overlapping labels
            for a in range(len(texts)):
                for b in range(a + 1, len(texts)):
                    if bbs[a].overlaps(bbs[b]):
                        moved = True
                        vx = centers[a][0] - centers[b][0]
                        vy = centers[a][1] - centers[b][1]
                        if vx == 0 and vy == 0:
                            # random tiny push if perfectly overlapped
                            vx, vy = rng.normal(size=2)
                        norm = np.hypot(vx, vy) + 1e-6
                        ux, uy = vx / norm, vy / norm
                        # nudge both in opposite directions
                        disp[a] += np.array([ux, uy]) * label_repel_pixels
                        disp[b] -= np.array([ux, uy]) * label_repel_pixels

            if not moved:
                break

            # apply pixel displacements in data space; optionally keep leader lines attached
            inv = ax.transData.inverted()
            for k, t in enumerate(texts):
                x, y = t.get_position()
                xpix, ypix = ax.transData.transform((x, y))
                xpix += disp[k, 0]
                ypix += disp[k, 1]
                x_new, y_new = inv.transform((xpix, ypix))
                t.set_position((x_new, y_new))
                if draw_leader_lines:
                    xa, ya = anchors[k]
                    lines[k].set_data([xa, x_new], [ya, y_new])

            # refresh extents every few iterations for accuracy
            if (it + 1) % 5 == 0:
                fig.canvas.draw()

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal", "datalim")
    if show_legend and handles:
        ax.legend(handles, legend_labels, frameon=False, fontsize=8, loc="best")

    plt.tight_layout()
    plt.savefig(savepath,dpi=400)


# ---- 3) Quick “best effort” all-in-one overview (faded others, ellipses + sparse labels) ----
def overview(Y, labels, top_labels=15, point_size=6):
    plt.figure(figsize=(40,40))
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    k = len(uniq)

    # color per cluster using HSV wheel (good enough for many classes)
    colors = plt.cm.hsv(np.linspace(0, 1, k, endpoint=False))
    mapping = {lab:i for i,lab in enumerate(uniq)}
    idxs = np.array([mapping[v] for v in labels], int)

    # draw background first for clarity

    for i, lab in enumerate(uniq):
        m = (idxs == i)
        plt.scatter(Y[m,0], Y[m,1], s=point_size, alpha=0.18, c=[colors[i]], linewidths=0)

    # annotate and ellipse only the largest few
    counts = np.array([(labels==lab).sum() for lab in uniq])
    big_idx = np.argsort(counts)[::-1][:min(top_labels, k)]
    from matplotlib.patches import Ellipse
    import numpy.linalg as LA

    for i in big_idx:
        m = (idxs == i)
        pts = Y[m]
        if pts.shape[0] < 5: continue
        cx, cy = pts.mean(axis=0)
        plt.text(cx, cy, f"{uniq[i]} ({pts.shape[0]})", ha="center", va="center", fontsize=8, weight="bold")
        cov = np.cov(pts.T); vals, vecs = LA.eigh(cov); o = vals.argsort()[::-1]
        vals, vecs = vals[o], vecs[:,o]
        ang = np.degrees(np.arctan2(vecs[1,0], vecs[0,0]))
        w, h = 2*np.sqrt(vals)  # 1σ ellipse
        plt.gca().add_patch(Ellipse((cx, cy), w, h, angle=ang, fill=False, lw=1, alpha=0.6))

    plt.title(f"{k} clusters (overview)")
    plt.xlabel("Component 1"); plt.ylabel("Component 2")
    plt.gca().set_aspect("equal","datalim"); plt.tight_layout(); plt.savefig('2.png',dpi=500)


# -------------------- How to use --------------------
# 1) Make a better 2D projection (tries supervised UMAP first)
Y, used = project_2d(embeddings, assign, method="auto")
print("Projection:", used)

# 2) Big-picture overview (faded points + ellipses + sparse labels)
overview(Y, assign, top_labels=15, point_size=5)

# 3) Drill-down: one-vs-all panels for the largest 16 clusters
focus_grid(Y, assign, top=16, cols=4, point_size=40, jitter=0)
