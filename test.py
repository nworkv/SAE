# -*- coding: utf-8 -*-
# Import 
from transformers import AutoModelForImageClassification
import copy

# Configuration
from conf import *

#####################################################################
# Instantiate the model
model = AutoModelForImageClassification.from_pretrained("WinKawaks/vit-small-patch16-224")
model.config.output_hidden_states = True
model.eval()
for param in model.parameters():
  param.requires_grad = False
model = model.to(DEVICE)

dim_model = model.config.hidden_size
hidden_dim = HIDDEN_MULTIPLIER * dim_model
sae = SparseAutoEncoder(dim_model, hidden_dim, SPARSITY_K).to(DEVICE)
sae = load_model(sae, LOAD_PATH)

model_modified = copy.deepcopy(model)
model_modified.vit.encoder.layer[LAYER_INDEX] = sae
model_modified = model_modified.to(DEVICE)
for param in model_modified.parameters():
  param.requires_grad = False
  
# --- Example image ---
import requests
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
inputs = processor(images=image, return_tensors="pt").to(DEVICE)

outputs = model(**inputs)
logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])

outputs = model_modified(**inputs)
logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])

probs = F.softmax(logits, dim=-1)
top5_probs, top5_labels = torch.topk(probs, k=5)
id2label = model.config.id2label

print("Top 5 Predictions (using SAE-reconstructed features):")
for prob, label_idx in zip(top5_probs[0], top5_labels[0]):
  label = id2label[label_idx.item()]
  print(f"{label}: {prob.item()*100:.2f}%")