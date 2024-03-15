import torch
import os
from collections import defaultdict
from PIL import Image
import json
from tqdm import tqdm
from transformers import LlavaForConditionalGeneration, LlavaProcessor
from argparse import ArgumentParser

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(53)

parser = ArgumentParser()
parser.add_argument("--cache_dir", type=str, default="hf_cache")
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


model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-13b-hf", 
    load_in_8bit=LOAD_QUANTIZED,
    low_cpu_mem_usage=True,
    bnb_4bit_compute_dtype=torch.float16,
    cache_dir=CACHE_DIR,
    torch_dtype=torch.float16
    ).eval()

processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-13b-hf", cache_dir=CACHE_DIR)
processor.tokenizer.padding_side = "left"




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

base_prompt = "USER: You'll be given an image, an instruction and some options. You have to select the correct one. Do not explain your reasoning. Answer with only the letter that corresponds to the correct option. Do not repeat the entire answer. <image>\n"


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
    prompt = base_prompt + data["question"] + "\n"+mcq+"\nASSISTANT:"
    
    
    image = Image.open(os.path.join(EVAL_IMAGE_DIR, data["image"]))
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, min_length=1, max_new_tokens=1)
        llava_answer = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        llava_answer = llava_answer.split("ASSISTANT: ")[-1][0].strip().lower()
    
    eval_dataset[i]["llava_answer"] = llava_answer

with open(os.path.join(SAVE_DIR, DATASET+"_llava_results.json"), "w") as f:
    json.dump(eval_dataset, f, indent=4)



