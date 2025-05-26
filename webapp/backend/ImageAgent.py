import json
from textwrap import dedent
from agno.tools import Toolkit
from agno.agent import Agent
from agno.models.ollama import Ollama
import torch
from transformers import AutoTokenizer, AutoModel
from agno.agent import Agent
from agno.models.ollama import Ollama
from duckduckgo_search import DDGS
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

import os
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
import shutil
# 1. Prepare model - ResNet50 without final fc layer
class FeatureExtractor(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        # Use all layers except final fc
        self.features = torch.nn.Sequential(*list(model.children())[:-1])
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        return x

# Load pretrained ResNet50
resnet = models.resnet50(pretrained=True)
res_model = FeatureExtractor(resnet)
res_model.eval()

# 2. Prepare image transform (resize, normalize as ImageNet)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]    # ImageNet std
    )
])

def extract_feature(image_path):
    img = Image.open(image_path).convert('RGB')
    img_t = transform(img)
    batch_t = img_t.unsqueeze(0)  # batch dimension
    with torch.no_grad():
        feat = res_model(batch_t)
    return feat

# 3. Extract features for all images in folder
def check_image_relevant():
    folder_path = '/home/veronrd/chatbot/VLMs/images/input'
    folder_images = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    folder_features = []
    for img_path in folder_images:
        feat = extract_feature(img_path)
        folder_features.append((img_path, feat))

    # 4. Extract feature for input image
    input_image_path = 'la1.jpg'
    input_feat = extract_feature(input_image_path)

    # 5. Compute similarity with each folder image feature (cosine similarity)
    def cosine_sim(a, b):
        return F.cosine_similarity(a, b).item()

    best_match = None
    best_score = -1  # cosine similarity ranges from -1 to 1
    for path, feat in folder_features:
        sim = cosine_sim(input_feat, feat)
        if sim > best_score:
            best_score = sim
            best_match = path
    best_label = best_match.replace("/input/", "/label/")
    best_label = best_label.replace(".jpg", ".png")
    destination = "la2.jpg"

    shutil.copyfile(best_label, destination)
    # print(f"Copied {new_path} to {destination}")



path = "OpenGVLab/InternVL2_5-1B"
vlm_model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    # load_in_8bit=True,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().to('cuda')
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

from pathlib import Path



generation_config = dict(max_new_tokens=1024, do_sample=False)

class ImageTool(Toolkit):
    # @validate_call
    def __init__(self, **kwargs):
        super().__init__(name="image_tools", **kwargs)
        self.register(self.image_overview)
        self.register(self.describe_image)
        self.register(self.check_leaf_disease)
        self.register(self.search_solutions)
    def image_overview(self) -> str:
        question = (
            "<image>\n"
            "Determine whether this image contains a tree leaf or any kind of plant leaf"
        )
        pixel_values = load_image('la1.jpg', max_num=12).to(torch.bfloat16).cuda()
        response = vlm_model.chat(tokenizer, pixel_values, question, generation_config)
        return response
    
    def describe_image(self) -> str:
        question = (
            "<image>\n"
            "Describe the content of this image in detail."
        )
        pixel_values = load_image('la1.jpg', max_num=12).to(torch.bfloat16).cuda()
        response = vlm_model.chat(tokenizer, pixel_values, question, generation_config)
        return response
    
    
    def check_leaf_disease(self) -> str:  
        pixel_values1 = load_image('la1.jpg', max_num=12).to(torch.bfloat16).cuda()
        check_image_relevant()
        pixel_values2 = load_image('la2.jpg', max_num=12).to(torch.bfloat16).cuda()
        pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
        num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]

        question = 'The original leaf image is: <image>\n The leaf image with disease segmentation is: <image>\nDescribe the leaf disease base on the original image and the leaf disease detect image.'
        response = vlm_model.chat(tokenizer, pixel_values, question, generation_config, num_patches_list=num_patches_list)
        return response
    
    def search_solutions(self, query: str, max_results: int = 5) -> str:
        actual_max_results = max_results
        search_query = query

        ddgs = DDGS(
        )
        return json.dumps(ddgs.text(keywords=search_query, max_results=actual_max_results), indent=2)

def get_image_agent():
    agent = Agent(
        model=Ollama(id="qwen3:4b"),
        tools=[ImageTool()],
        instructions=dedent("""\
        You are an image analysis assistant with expertise in identifying leaf-related features and detecting potential diseases when present. Use the image provided by the tool to guide your analysis.

        1. First, assess whether the image contains a leaf and whether the **main focus** is on the leaf or a leaf disease.
        2. If the image is **not primarily focused** on a leaf or leaf disease, clearly state that and provide a detailed and accurate description of the image's main subject.
        3. If the image **is primarily focused** on a leaf, examine it for any visible signs of disease or abnormality.
        4. If signs of disease are found, suggest potential causes and treatments by conducting a search for relevant solutions.
    """),
        markdown=True,
    )
    return agent




