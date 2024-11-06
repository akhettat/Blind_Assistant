#This script aims to download a  model from huggingface
from huggingface_hub import snapshot_download
from pathlib import Path

# Chemin pour stocker le modèle localement
model_dir = Path.home().joinpath('mistral_models', '7B-v0.1')
print(model_dir)
model_dir.mkdir(parents=True, exist_ok=True)

# Télécharger le modèle avec tous ses fichiers
snapshot_download(repo_id="mistralai/Mistral-7B-v0.1", local_dir=model_dir, allow_patterns=["*"])
