import torch 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pdb
activations = torch.load('./data/activations_sae.pt')


scaler = StandardScaler()

transformed = scaler.fit_transform(activations)

X = torch.tensor(transformed, dtype=torch.float32, device="cuda")

# Center the data (subtract mean per feature)
X = X - X.mean(dim=0)

# Compute covariance matrix
cov = X.T @ X / (X.shape[0] - 1)

# Eigen-decomposition on GPU
eigenvalues, eigenvectors = torch.linalg.eigh(cov)

# Sort in descending order
idx = torch.argsort(eigenvalues, descending=True)
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Explained variance ratio
explained_variance_ratio = eigenvalues / eigenvalues.sum()


nums = list(range(5, 1000))
variations = [explained_variance_ratio[:num].sum().item() for num in nums]


plt.plot(nums,variations)
plt.savefig('1.png')
