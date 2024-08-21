import torch
import os
from collections import defaultdict
from PIL import Image
import json
from tqdm import tqdm
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
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




path = "microsoft/Phi-3.5-vision-instruct" 
model = AutoModelForCausalLM.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    load_in_8bit=LOAD_QUANTIZED,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    cache_dir=CACHE_DIR).to(device).eval()


processor = AutoProcessor.from_pretrained(path, 
  trust_remote_code=True, 
  num_crops=16
) 

generation_args = { 
    "max_new_tokens": 1000, 
    "temperature": 0.0, 
    "do_sample": False, 
} 

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

base_prompt = "<|image_1|>\n {question}\n{mcq}\nAnswer with only the letter that corresponds to the correct option. Do not repeat the entire answer. Do not explain your reasoning."


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
    messages = [
        {"role": "user", "content": prompt},
    ]
    prompt = processor.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    image = Image.open(os.path.join(EVAL_IMAGE_DIR, data["image"])).convert('RGB')

    inputs = processor(prompt, image, return_tensors="pt").to("cuda:0") 


    generate_ids = model.generate(**inputs, 
    eos_token_id=processor.tokenizer.eos_token_id, 
    **generation_args
    )

    # remove input tokens 
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids, 
    skip_special_tokens=True, 
    clean_up_tokenization_spaces=False)[0] 

    print(response)
    
    eval_dataset[i]["phi35_answer"] = response

with open(os.path.join(SAVE_DIR, DATASET+"_phi35_results.json"), "w") as f:
    json.dump(eval_dataset, f, indent=4)



