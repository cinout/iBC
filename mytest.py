import torchvision.models as models

# Load ViT-B/16
model = models.vit_b_16(weights=None)

# Print full architecture
print(model)
