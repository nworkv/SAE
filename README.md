# SAE
SAE for VIT

# SRGAN
The research investigates how Sparse Autoencoders can enhance the understanding of internal representations within Vision Transformers. <br>
Notebook : [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zHiTLdVccE1zszPuBlcO4H7k8jPk8tgN?usp=sharing)

## Project goal
*   Train Sparse Autoencoders on one of layers from a pre-trained Vision Transformer (ViT-Tiny).
*   Investigate the capabilities of the SAE you’ve trained by discovering which images and classes result in the maximum activation of specific neurons in the network.
*   Analyze the interpretability of the learned features.
## Components

*   **`conf.py`**: Configuration file.
    ```
    IMAGE_FOLDER ='/kaggle/input/flickr2k/Flickr2K'
    OUTPUT_DIR = "/content/gdrive/MyDrive/SparceAutoEncoder"
    LOAD_PATH = "/content/gdrive/MyDrive/SparceAutoEncoder/sae_20250712_104941"  
    BATCH_SIZE = 30
    NUM_EPOCHS = 10
    LR = 0.002
    SHEDULER_LAMBDA_PARAM = 0.7
    LAYER_INDEX = 6
    HIDDEN_MULTIPLIER = 16         # Latent dimension = 16 × input_dim
    SPARSITY_K = 64                # Top-K sparsity
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ```
*   **`dsdl.py`**: Loads an ImageDataset / ImageDataloader. It is nessasary in order to properly prepare the training data.
*   **`sparcesutoencoder.py`**: Loads a SparseAtuEncoder class.
*   **`training.py`**: Imports training functhion that starts the training process with nessasary 
*   **`test.py`**: Demonstrates how to use the trained SAE to analize neurons.

## Setup
1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    ```
2.  **Install dependencies:**
    ```bash
    pip install torch torchvision Pillow tqdm request transformers # Add other dependencies if needed
3.  **Prepare Dataset:** Download and structure your image dataset. Update the paths in the scripts (`BASE_IMAGE_FOLDER`, `IMAGE_FOLDER`, ....) accordingly.
4.  **Prepare SAE:** Update the path in th script (`LOAD_PATH`) where the model is located.

## Architecture of SAE
```
class SparseAutoEncoder(nn.Module):
  def __init__(self, input_dim: int, hidden_dim: int, k: int):
    super(SparseAutoEncoder, self).__init__()

    self.layernorm_before = nn.LayerNorm(input_dim , eps=1e-12)
    #self.layernorm_after = nn.LayerNorm(input_dim , eps=1e-12)
    self.encoder = nn.Linear(input_dim, hidden_dim)
    self.latent_bias = nn.Parameter(torch.zeros(hidden_dim))
    self.act = nn.ReLU()
    self.decoder = nn.Linear(hidden_dim, input_dim)
    self.k = k

    with torch.no_grad():
      self.decoder.weight.copy_(self.encoder.weight.T)
      self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True) + 1e-8)

  def forward(self, input, head_mask=None, output_attentions=False):
    # head_mask=None, output_attentions=False - was added to smoothly integrate SparseAutoEncoder in Vit

    latent = self.get_latent(input)
    recon = self.decoder(latent)
    return (recon, head_mask, output_attentions)

  def get_latent(self, input):
    input = self.layernorm_before(input)
    latent_pre = self.encoder(input) + self.latent_bias
    vals, idxs = torch.topk(latent_pre, self.k, dim=1)
    latent = torch.zeros_like(latent_pre).scatter(1, idxs, vals)
    latent = self.act(latent)
    return latent
```

