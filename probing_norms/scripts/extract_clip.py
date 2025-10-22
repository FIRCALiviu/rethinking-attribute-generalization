import open_clip
import torch
import pdb 
from tqdm import tqdm
from torch.utils.data import TensorDataset
from probing_norms.data import DATASETS
import torchvision.transforms as transforms



torch.set_float32_matmul_precision("high")

clip = "openai/ViT-B-16"

clip_version, clip_architecture = clip.split("/")

model, _, preprocess = open_clip.create_model_and_transforms(
    clip_architecture, pretrained=clip_version
)
model = model.bfloat16()
model.eval()
device = torch.device("cuda")
model.to(device)



train_ds = DATASETS['things'](transform= preprocess) 
train_dl = torch.utils.data.DataLoader(
    train_ds,
    batch_size=512,
    shuffle=False,  
    num_workers=10,
    pin_memory=True,
)

all_embeddings = []
get_classes = []
all_filepath = []
for images in tqdm(train_dl):
    with torch.no_grad():
        output = model.encode_image(images['image'].to(device=device, dtype=torch.bfloat16))
    all_filepath.extend(list(map(lambda x : x.split(r'/')[-1],images['name'])))

    all_embeddings.append(output.detach().cpu())
    get_classes.extend(images['label'])
all_embeddings = torch.cat(all_embeddings, 0)

torch.save(all_embeddings,'./data/all_embeddings.pt')

with open('./data/labels_images.txt','w') as f :
    f.write('\n'.join(map(str,get_classes)))

with open('./data/all_filepath.txt','w') as f :
    f.write("\n".join(all_filepath))