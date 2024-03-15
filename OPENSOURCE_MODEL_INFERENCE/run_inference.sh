#expects illusionVQA/comprehension and illusionVQA/sofloc to be present in the current directory

mkdir results
mkdir hf_cache

ENV_NAME="illusions_env"

if ! conda env list | grep -q "\b${ENV_NAME}\b"; then
  conda create -n ${ENV_NAME} python=3.11 -y 
fi

conda activate ${ENV_NAME} || source activate ${ENV_NAME}

pip install -q torch torchvision torchaudio bitsandbytes accelerate transformers sentencepiece protobuf einops xformers spacy

python cogvlm_inference.py --load_quantized False --dataset comprehension
python cogvlm_inference.py --load_quantized False --dataset sofloc

python llava15_inference.py --load_quantized False --dataset comprehension
python llava15_inference.py --load_quantized False --dataset sofloc

python instructblip_inference.py --load_quantized False --dataset comprehension
python instructblip_inference.py --load_quantized False --dataset sofloc

#zip results
zip -r results.zip results
