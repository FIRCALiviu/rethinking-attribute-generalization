import torch
from probing_norms.models import SAE,get_sae_pre_activations
import pdb 

DEVICE = 'cuda:0'
batch_size = 64
sae = SAE(512, 16384, 2)

# CONSTANT scalar for embeddings
ct_st = torch.load('./data/model.pt', map_location=DEVICE, weights_only=False)
dataset_scalar = ct_st["dataset_scalar"].item()

sae_st = torch.load('./data/5.pt', map_location=DEVICE, weights_only=True)
for key in list(sae_st.keys()):
    sae_st[key.replace("_orig_mod.", "")] = sae_st.pop(key)

sae.load_state_dict(sae_st)

embeddings = torch.load('./data/all_embeddings.pt')

activations = get_sae_pre_activations(sae,dataset_scalar,embeddings,DEVICE,64)


torch.save(activations,'./data/activations_sae.pt')