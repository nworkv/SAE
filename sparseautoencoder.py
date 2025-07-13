import torch.nn as nn
import torch
import os
from datetime import datetime
from conf import DEVICE
###################################################################################
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
######################################################################################

#saving model fuctions
def save_model(model, folder_path, name):
  timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
  PATH = os.path.join(folder_path, name  + '_{}'.format(timestamp))
  torch.save(model.state_dict(), PATH)

def create_chekpoint(model, optimizer, criterion, scheduler, epoch, folder_path):
  FORMAT = ".chk"
  NAME = "CHEKPOINT"
  timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
  PATH = os.path.join(folder_path, NAME + "_e" + str(epoch) + '_{}_'.format(timestamp) + FORMAT)
  torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'criterion': criterion
            }, PATH)
  
# Load model function
def load_model(model, PATH):
  state_dict = torch.load(PATH, weights_only=True, map_location=torch.device(DEVICE))
  model.load_state_dict(state_dict)
  return model