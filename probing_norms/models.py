import torch
from tqdm import tqdm

def rectangle(x):
    return ((x > -0.5) & (x < 0.5)).type(x.dtype)


class JumpReLUF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        result = x * (x > threshold).type(x.dtype)
        ctx.save_for_backward(x, threshold)
        ctx.stepf_bandwidth = bandwidth
        return result

    @staticmethod
    def backward(ctx, grad_output):
        # print('\ngrad output')
        # print(grad_output.shape)
        # print(grad_output.std())
        x, threshold = ctx.saved_tensors
        x_grad = (x > threshold) * grad_output
        bandwidth = ctx.stepf_bandwidth
        threshold_grad = -(threshold / bandwidth) * (
            rectangle((x - threshold) / bandwidth) * grad_output
        ).mean(0)
        # print()
        # print(threshold_grad.shape)
        # print(threshold_grad.norm(2))
        # print()
        return x_grad, threshold_grad, None


class JumpReLU(torch.nn.Module):
    def __init__(self, bandwidth):
        super().__init__()
        self.bandwidth = bandwidth

    def __call__(self, x, threshold):
        return JumpReLUF.apply(x, threshold, self.bandwidth)


class SAE(torch.nn.Module):
    def __init__(self, n, m, epsilon, pre_enc_bias=True):
        super().__init__()

        self.encoder = torch.nn.Linear(n, m)
        self.jumprelu = JumpReLU(bandwidth=epsilon)
        self.log_threshold = torch.nn.Parameter(torch.empty(m, dtype=torch.float16))
        self.decoder = torch.nn.Linear(m, n)
        self.pre_enc_bias = pre_enc_bias

        torch.nn.init.constant_(self.log_threshold, 0.1)
        torch.nn.init.uniform_(self.decoder.weight, -1.0 / n, 1.0 / n)
        torch.nn.init.zeros_(self.decoder.bias)
        with torch.no_grad():
            self.encoder.weight.copy_(self.decoder.weight.T)
            self.encoder.weight *= n / m
        torch.nn.init.zeros_(self.encoder.bias)

    def forward(self, x):
        if self.pre_enc_bias:
            x = x - self.decoder.bias
        pre_activations = self.encoder(x)
        relu_pre_activations = torch.nn.functional.relu(pre_activations)
        threshold = torch.exp(self.log_threshold)
        features = self.jumprelu(relu_pre_activations, threshold)
        # features = relu_pre_activations
        reconstruction = self.decoder(features)
        return reconstruction, features, pre_activations
    
    def get_features(self, x):
        if self.pre_enc_bias:
            x = x - self.decoder.bias
        
        pre_activations = self.encoder(x)
        relu_pre_activations = torch.nn.functional.relu(pre_activations)
        threshold = torch.exp(self.log_threshold)
        features = self.jumprelu(relu_pre_activations, threshold)
        return features

    def get_pre_activations(self, x):
        if self.pre_enc_bias:
            x = x - self.decoder.bias
        pre_activations = self.encoder(x)
        return pre_activations
    
        

    def fwd_with_zero_mask(self, x, mask, return_features=False):
        x_ = x
        if self.pre_enc_bias:
            x_ = x - self.decoder.bias

        threshold = torch.exp(self.log_threshold)
        pre_activations = self.encoder(x_)
        features = self.jumprelu(pre_activations, threshold)
        features_to_remove = features * mask
        decoded_activations_to_remove = (
            self.decoder(features_to_remove) - self.decoder.bias
        )
        sae_reconstruction = x - decoded_activations_to_remove
        if return_features:
            return sae_reconstruction, features
        return sae_reconstruction

        # if self.pre_enc_bias:
        #     x = x - self.decoder.bias
        # pre_activations = self.encoder(x)
        # relu_pre_activations = torch.nn.functional.relu(pre_activations)
        # # print("relu_pre_activations", relu_pre_activations.shape)
        # threshold = torch.exp(self.log_threshold)
        # features = self.jumprelu(relu_pre_activations, threshold)
        # # print("features", features.shape)
        # # features = relu_pre_activations
        # features[:, self.TO_ZERO_INDICES_256] = 0
        # reconstruction = self.decoder(features)
        # return reconstruction, features, pre_activations



@torch.no_grad
def get_sae_pre_activations(
    sae_model,
    ds_scalar,
    input_embeddings,
    device,
    bs,
    num_workes=8,
):
    dl = torch.utils.data.DataLoader(
        input_embeddings,
        batch_size=bs,
        shuffle=False,
        drop_last=False,
        num_workers=num_workes,
    )

    sae_model.eval()
    sae_model.to(device)
    features_16k = []

    for batch in tqdm(dl):
        features = sae_model.get_pre_activations(batch.to(device) * ds_scalar)
        features_16k.append(features.cpu())

    features_16k = torch.cat(features_16k, dim=0)
    return features_16k