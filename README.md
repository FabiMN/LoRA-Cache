# LoRA-Cache: Low-Rank Few-Shot Adaptation
This repository contains code that was written as part of my masters thesis.

## Installation
```bash
git clone https://github.com/FabiMN/LoRA-Cache.git
cd LoRA-Cache

conda create -n lora_cache python=3.8
conda activate lora_cache

pip install -r requirements.txt

conda install pytorch torchvision cudatoolkit
```

## Running the code
Run main.py to run the main approach of LoRA-Cache.
Run main_bottleneck_adapter.py to run the approach that injects bottleneck adapters into the first linear layer of each MLP modules.
Run main_mha to run the approach that injects low-rank decomposition matrices into various linear layers in the self-attention modules.

As an example, to run main.py using the Caltech101 dataset, run the following command:
```bash
python main.py --config configs/caltech101.yaml
```

## Acknowledgement
This repository is build on top of prior work done by [Tip-Adapter](https://github.com/gaopengcuhk/Tip-Adapter),  [CLIP](https://github.com/openai/CLIP), [CoOp](https://github.com/KaiyangZhou/Dassl.pytorch) and [CLIP-Adapter](https://github.com/gaopengcuhk/CLIP-Adapter).

