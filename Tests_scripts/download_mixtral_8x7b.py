from huggingface_hub import snapshot_download
import logging

logging.basicConfig(level=logging.DEBUG)

huggingface_token = "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"

# Spécifiez le dépôt du modèle et le répertoire de téléchargement
repo_id = "mistralai/Mixtral-8x7B-v0.1"
model_dir = "./mixtral_8x7b"  # Répertoire où le modèle sera téléchargé

# Téléchargement du modèle
print("Téléchargement de Mixtral 8x7B en cours...")
snapshot_download(
    repo_id=repo_id,
    local_dir=model_dir,
    use_auth_token=huggingface_token,  # Utilisez le nouveau token
    allow_patterns=["*"]
)

print("Téléchargement terminé ! Le modèle est disponible dans", model_dir)
