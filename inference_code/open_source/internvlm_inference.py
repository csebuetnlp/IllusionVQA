import torch
import os
from collections import defaultdict
from PIL import Image
import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from argparse import ArgumentParser

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(53)

parser = ArgumentParser()
parser.add_argument("--cache_dir", type=str, default="my_cache")
parser.add_argument("--base_dir", type=str, default="illusionVQA")
parser.add_argument("--save_dir", type=str, default="results")
parser.add_argument("--dataset", type=str)
parser.add_argument("--load_quantized", type=bool, default=False)

args = parser.parse_args()

CACHE_DIR = args.cache_dir
BASE_DIR = args.base_dir
DATASET = args.dataset
SAVE_DIR = args.save_dir
LOAD_QUANTIZED = args.load_quantized


os.environ["HF_HOME"]=CACHE_DIR
EVAL_JSON = os.path.join(BASE_DIR, DATASET, "eval_labels.json")
EVAL_IMAGE_DIR = os.path.join(BASE_DIR, DATASET, "EVAL")




path = "OpenGVLab/InternVL2-8B"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    load_in_8bit=LOAD_QUANTIZED,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    cache_dir=CACHE_DIR).to(device).eval()

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)


import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

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


generation_config = dict(max_new_tokens=1024, do_sample=False)


 

def construct_mcq(options, correct_option):
    correct_option_letter = None
    i = "a"
    mcq = ""

    for option in options:
        if option == correct_option:
            correct_option_letter = i
        mcq += f"{i}. {option}\n"
        i = chr(ord(i) + 1)

    mcq = mcq[:-1]
    return mcq, correct_option_letter

base_prompt = "<image>\n {question}\n{mcq}\nAnswer with only the letter that corresponds to the correct option. Do not repeat the entire answer. Do not explain your reasoning."


with open(EVAL_JSON) as f:
    eval_dataset = json.load(f)

category_count = defaultdict(int)
import os
for data in eval_dataset:
    if data["image"] not in os.listdir(EVAL_IMAGE_DIR):
        print(data["image"])
        continue
    data["image_path"] = EVAL_IMAGE_DIR + data["image"]
    data["mcq"], data["correct_option_letter"] = construct_mcq(data["options"], data["answer"])
    category_count[data["category"]] += 1


for i,data in tqdm(enumerate(eval_dataset)):

    mcq, answer = construct_mcq(data["options"], data["answer"])
    # prompt = base_prompt + data["question"] + "\n"+mcq+"\nASSISTANT:"
    
    prompt = base_prompt.format(question=data["question"], mcq=mcq)

    pixel_values = load_image(os.path.join(EVAL_IMAGE_DIR, data["image"]), max_num=12).to(torch.bfloat16).cuda()

    intern_answer = model.chat(tokenizer, pixel_values, prompt, generation_config)
    
    eval_dataset[i]["intern_answer"] = intern_answer

with open(os.path.join(SAVE_DIR, DATASET+"_intern_results.json"), "w") as f:
    json.dump(eval_dataset, f, indent=4)



