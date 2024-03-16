# IllusionVQA: Optical Illusion Dataset

Paper Link: <br>
Comprehension Dataset:     [link](https://huggingface.co/datasets/csebuetnlp/illusionVQA-Comprehension) <br>
Soft-Localization Dataset: [link](https://huggingface.co/datasets/csebuetnlp/illusionVQA-Soft-Localization)<br>

## TL;DR
IllusionVQA is a dataset of optical illusions and hard-to-interpret scenes, to test the capability of Vision Language Models in comprehension and soft localization tasks. GPT4V achieved 62.99% accuracy on comprehension and 49.7% on localization, while humans achieved 91.03% and 100% respectively.
## Usage
```
from datasets import load_dataset
import base64
from openai import OpenAI
import os
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

def encode_image(pil_image):
    temp_name = "temp.jpg"
    pil_image.save(temp_name)
    with open(temp_name, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

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

def add_row(content, data, i, with_answer=False):  

    mcq, correct_option_letter = construct_mcq(data["options"], data["answer"])

    content.append({
            "type": "text",
            "text": "Image "+str(i)+": "+data["question"]+"\n"+mcq
        })
    
    content.append(
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image(data["image"])}",
                "detail": "low"
            }
        }
    )
    if with_answer:
        content.append(
            {
                "type": "text",
                "text": "Answer {}: ".format(i)+correct_option_letter
            }
        )
    else:
        content.append(
            {
                "type": "text",
                "text": "Answer {}: ".format(i),
            }
        )
    
    return content

dataset = load_dataset("csebuetnlp/illusionVQA-Comprehension")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


content = [
    {
        "type": "text",
        "text": "You'll be given an image, an instruction and some choices. You have to select the correct one. Do not explain your reasoning. Answer with the option's letter from the given choices directly. Here are a few examples:",
    }
]

### Add the few examples
i = 1
for data in dataset["train"]:
    content = add_row(content, data, i, with_answer=True)
    i += 1

content.append({
                    "type": "text",
                    "text": "Now you try it!",
                })

next_idx = i

### Add the test data
test_data = dataset["test"][0]
content_t = add_row(content.copy(), test_data, next_idx, with_answer=False)

### Get the answer from GPT-4
response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user",
            "content": content_t,
        }
    ],
    max_tokens=5,
)
gpt4_answer = response.choices[0].message.content
print(gpt4_answer)
```

## License
This dataset is made available for non-commercial research purposes only, including for evaluation of model performance. The dataset may not be used for training models. The dataset contains images collected from the internet. While permission has been obtained from some of the images' creators, permission has not yet been received from all creators. If you believe any image in this dataset is used without proper permission and you are the copyright holder, please get in touch with sameen2080@gmail.com to request the removal of the image from the dataset.

The dataset creator makes no representations or warranties regarding the copyright status of the images in the dataset. Use of this dataset is at your own risk. The dataset creator shall not be held liable for any unauthorized use of copyrighted material that may be contained in the dataset.

You agree to the terms and conditions specified in this license by downloading or using this dataset. If you do not agree with these terms, do not download or use the dataset.


### Citation
