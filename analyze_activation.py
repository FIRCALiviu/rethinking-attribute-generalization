import torch 
import pdb
neuron = 10470

activations  = torch.load('./data/activations_sae.pt')
filenames = open('./data/all_filepath.txt').read().split()

threshold = activations[:,neuron].std()+activations[:,neuron].mean()
print(threshold.item())
positive = activations[:,neuron] > threshold
with open('positive.txt','w') as f:
    f.write("\n".join([filenames[i] for i in range(len(positive)) if positive[i]]))

