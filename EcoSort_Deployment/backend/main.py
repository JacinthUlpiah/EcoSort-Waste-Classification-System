from fastapi import FastAPI, UploadFile, File
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io

app = FastAPI(title="EcoSort API")

# Use CPU for the Dockerized API to keep the container lightweight
device = torch.device("cpu") 

# Rebuild model structure and load weights
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load("models/ecosort_efficientnet_b0.pth", map_location=device))
model.eval()

# Strict testing transforms (no augmentation)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

classes = ["Organic", "Recyclable"]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)
        
    return {"filename": file.filename, "prediction": classes[predicted.item()]}