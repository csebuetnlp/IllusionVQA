from transformers import BitsAndBytesConfig
from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
)
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
)
from llava.conversation import conv_templates
from PIL import Image
import copy
import torch
import warnings
import os
import json
from collections import defaultdict
from tqdm import tqdm
from argparse import ArgumentParser


torch.manual_seed(53)

parser = ArgumentParser()
parser.add_argument("--cache_dir", type=str, default="hf_cache")
parser.add_argument("--base_dir", type=str, default="illusionVQA")
parser.add_argument("--save_dir", type=str, default="../../result_jsons")
parser.add_argument("--dataset", type=str, default="sofloc")
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

warnings.filterwarnings("ignore")
pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
tokenizer, model, image_processor, max_length = load_pretrained_model(
    pretrained,
    None,
    model_name,
    device_map=device_map,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
)  # Add any other thing you want to pass in llava_model_args
model.eval()


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

for data in eval_dataset:
    if data["image"] not in os.listdir(EVAL_IMAGE_DIR):
        print(data["image"])
        continue
    data["image_path"] = EVAL_IMAGE_DIR + data["image"]
    data["mcq"], data["correct_option_letter"] = construct_mcq(
        data["options"], data["answer"]
    )
    category_count[data["category"]] += 1


conv_template = "qwen_1_5"

for i, data in enumerate(tqdm(eval_dataset)):
    try:
        mcq, answer = construct_mcq(data["options"], data["answer"])
        prompt = base_prompt + data["question"] + "\n" + mcq + "\nASSISTANT:"

        image = Image.open(os.path.join(EVAL_IMAGE_DIR, data["image"]))
        image_sizes = [image.size]  # type: ignore
        image = process_images([image], image_processor, model.config)
        image = [img.to(dtype=torch.float16, device=device) for img in image]

        prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)  # type: ignore
            .to(device)
        )

        outputs = (
            model.generate(
                input_ids,
                images=image,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                max_new_tokens=4096,
            )
            .detach()
            .cpu()
        )
        llava_answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        llava_answer = llava_answer.split("ASSISTANT: ")[-1][0].strip().lower()
        eval_dataset[i]["llava_onevision_ans"] = llava_answer
    except Exception as e:
        print(e)
        print("skipping", i)
        torch.cuda.empty_cache()

with open(
    os.path.join(SAVE_DIR, f"llava_onevision_qwen2_7b_{DATASET}_0shot.json"), "w"
) as f:
    json.dump(eval_dataset, f, indent=4)
