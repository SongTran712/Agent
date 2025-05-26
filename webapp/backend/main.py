from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from agno.agent import Agent, RunResponse
from agno.workflow import RunResponse
from CSVAgent import get_csv_agent
import os
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import shutil
from ImageAgent import get_image_agent, load_image
folder_path = "/home/veronrd/chatbot/VLMs/images/input"
input_image_path = "./la1.jpg"
import re
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting FastAPI application...")
    yield

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained feature extractor (ResNet50)
sim_model = models.resnet50(pretrained=True)
sim_model = torch.nn.Sequential(*list(sim_model.children())[:-1])  # remove final classification layer
sim_model.eval().to(device)

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet means
        std=[0.229, 0.224, 0.225]    # ImageNet stds
    )
])

def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = sim_model(image_tensor).squeeze()
        features = F.normalize(features, dim=0)  # normalize for cosine similarity
    return features
  
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class Item(BaseModel):
    content: str = Field(...)
    user: str = Field(...)
    sessionID: str = Field(...)
    
async def data_stream(llm, content):
    run_response = await llm.arun(content, stream=True)
    try:
        async for response in run_response:
            if isinstance(response, RunResponse):
                # print(response)
                response_content = response.content
                if response_content is not None:
                    print(response_content)
                    print('\n')
                    yield response_content
    finally:
        print("Stream success")

def get_most_similarity(image):
    input_feat = extract_features(input_image_path)

    # Compare with images in folder
    similarities = []
    image_files = []

    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            file_path = os.path.join(folder_path, file_name)
            feat = extract_features(file_path)
            sim = F.cosine_similarity(input_feat, feat, dim=0).item()
            similarities.append(sim)
            image_files.append(file_path)

    # Find the most similar image
    most_similar_index = int(np.argmax(similarities))
    most_similar_image = image_files[most_similar_index]
    return most_similar_image, most_similar_image.replace("/home/veronrd/chatbot/VLMs/images/input", "/home/veronrd/chatbot/VLMs/images/label")

@app.post("/api/chat")
async def ask(req: Item):
    csv_agent =  get_csv_agent()
    generator = data_stream(csv_agent, req.content)
    return StreamingResponse(generator, media_type="text/event-stream"
                        , headers={"cache-Control": "no-cache", "cf-cache-status": "DYNAMIC",
                                   "x-content-type-options": "nosniff", "content-type":"text/event-stream"}
                        )

async def image_data_stream(llm):
    run_response = llm.run()
    yield re.sub(r'<think>.*?</think>', '', run_response.content, flags=re.DOTALL).strip()
    # finally:
    #     print("Stream success")
        
@app.post("/api/image")
async def upload_image(file: UploadFile = File(...)):
    
    file_path = input_image_path
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    image_agent = get_image_agent()
    generator = image_data_stream(image_agent)
    return StreamingResponse(generator, media_type="text/event-stream"
                        , headers={"cache-Control": "no-cache", "cf-cache-status": "DYNAMIC",
                                   "x-content-type-options": "nosniff", "content-type":"text/event-stream"}
                        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)