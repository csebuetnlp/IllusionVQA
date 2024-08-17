import torch
from PIL import Image
from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
import os
from collections import defaultdict
from tqdm import tqdm
import json
from argparse import ArgumentParser

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(53)

parser = ArgumentParser()
parser.add_argument("--cache_dir", type=str, default="hf_cache")
parser.add_argument("--base_dir", type=str, default="illusionVQA")
parser.add_argument("--save_dir", type=str, default="results")
parser.add_argument("--dataset", type=str, default="comprehension")
parser.add_argument("--load_quantized", type=bool, default=False)

args = parser.parse_args()

CACHE_DIR = args.cache_dir
BASE_DIR = args.base_dir
DATASET = args.dataset
SAVE_DIR = args.save_dir
LOAD_QUANTIZED = args.load_quantized


os.environ["HF_HOME"] = CACHE_DIR
EVAL_JSON = os.path.join(BASE_DIR, DATASET, "eval_labels.json")
EVAL_IMAGE_DIR = os.path.join(BASE_DIR, DATASET, "EVAL")

MODEL_PATH = "google/paligemma-3b-mix-448"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_TYPE = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    else torch.float16
)

model = PaliGemmaForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    cache_dir=CACHE_DIR,
    device_map=DEVICE,
    torch_dtype=TORCH_TYPE,
    revision="bfloat16",
    load_in_4bit=LOAD_QUANTIZED,
).eval()
processor = AutoProcessor.from_pretrained(MODEL_PATH)


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


base_prompt = "answer en "

with open(EVAL_JSON) as f:
    eval_dataset = json.load(f)

category_count = defaultdict(int)

for data in eval_dataset:
    if data["image"] not in os.listdir(EVAL_IMAGE_DIR):
        print(data["image"])
        continue
    data["image_path"] = EVAL_IMAGE_DIR + data["image"]
    data["mcq"], data["correct_option_letter"] = construct_mcq(
        data["options"], data["answer"]
    )
    category_count[data["category"]] += 1

for i, data in enumerate(tqdm(eval_dataset)):
    mcq, answer = construct_mcq(data["options"], data["answer"])
    prompt = (
        base_prompt + data["question"] + "\n" + mcq + "\nSelect the option letter only:"
    )

    image = Image.open(os.path.join(EVAL_IMAGE_DIR, data["image"])).convert("RGB")
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        generation = generation[0][input_len:]
        decoded = processor.decode(generation, skip_special_tokens=True)

    eval_dataset[i]["gemma_answer"] = decoded

SAVE_PATH = os.path.join(SAVE_DIR, DATASET + "_paligemma_results.json")
os.makedirs(SAVE_DIR, exist_ok=True)

with open(SAVE_PATH, "w") as f:
    json.dump(eval_dataset, f, indent=4)
