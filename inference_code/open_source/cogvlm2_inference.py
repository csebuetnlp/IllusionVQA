import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
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
parser.add_argument("--load_quantized", type=bool, default=True)

args = parser.parse_args()

CACHE_DIR = args.cache_dir
BASE_DIR = args.base_dir
DATASET = args.dataset
SAVE_DIR = args.save_dir
LOAD_QUANTIZED = args.load_quantized


os.environ["HF_HOME"] = CACHE_DIR
EVAL_JSON = os.path.join(BASE_DIR, DATASET, "eval_labels.json")
EVAL_IMAGE_DIR = os.path.join(BASE_DIR, DATASET, "EVAL")


MODEL_PATH = "THUDM/cogvlm2-llama3-chat-19B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_TYPE = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    else torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, trust_remote_code=True, cache_dir=CACHE_DIR
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=TORCH_TYPE,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    cache_dir=CACHE_DIR,
    load_in_4bit=LOAD_QUANTIZED,
).eval()


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


base_prompt = "You'll be given an image, an instruction and some options. You have to select the correct one. Do not explain your reasoning. Answer with only the letter that corresponds to the correct option. Do not repeat the entire answer.\n"


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
    prompt = base_prompt + data["question"] + "\n" + mcq + "\nASSISTANT:"

    image = Image.open(os.path.join(EVAL_IMAGE_DIR, data["image"])).convert("RGB")
    inputs = model.build_conversation_input_ids(
        tokenizer, query=prompt, history=[], images=[image]
    )

    inputs = {
        "input_ids": inputs["input_ids"].unsqueeze(0).to("cuda"),
        "token_type_ids": inputs["token_type_ids"].unsqueeze(0).to("cuda"),
        "attention_mask": inputs["attention_mask"].unsqueeze(0).to("cuda"),
        "images": [[inputs["images"][0].to("cuda").to(torch.bfloat16)]],
    }
    with torch.no_grad():
        outputs = model.generate(**inputs, min_length=1, max_new_tokens=1)
        outputs = outputs[:, inputs["input_ids"].shape[1] :]
        cog_answer = tokenizer.decode(outputs[0])

    eval_dataset[i]["cog_answer"] = cog_answer


SAVE_PATH = os.path.join(SAVE_DIR, DATASET + "_cog2_results.json")
os.makedirs(SAVE_DIR, exist_ok=True)

with open(SAVE_PATH, "w") as f:
    json.dump(eval_dataset, f, indent=4)
