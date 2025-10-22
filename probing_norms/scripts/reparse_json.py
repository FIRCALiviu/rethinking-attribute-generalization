import json 
import pdb
name = 'train_test_split.json'
new_name = 'categs_split.json'
with open(name,'r') as f:
    data = json.load(f)
    new_object = {k:{'train_idxs':v[0],'test_idxs':v[1]} for k,v in zip(data['features'],data['splits'])}

    json.dump(new_object,open(new_name,'w'))