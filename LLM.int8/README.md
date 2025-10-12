---
language: en
license: apache-2.0
tags:
- quantization
- llm.int8
- bitsandbytes
- opt
- facebook
- pytorch
- transformers
- causal-lm
datasets:
- hellaswag
- piqa
- arc_easy
- arc_challenge
- openbookqa
- winogrande
- super-glue-lm-eval-v1
- wikitext
metrics:
- accuracy
- perplexity
---

# LLM.int8 Quantized OPT Models

This repository contains experiments and implementations of LLM.int8 quantization using BitsAndBytes for OPT (Open Pre-trained Transformer) models. LLM.int8 is a quantization method that converts model weights to 8-bit precision while maintaining high accuracy through mixed-precision inference.

## Model Details

### Model Description

These models are quantized versions of Facebook's OPT (Open Pre-trained Transformer) models using the LLM.int8 quantization method. The quantization preserves model performance while significantly reducing memory requirements.

- **Developed by:** YuvrajSingh-mist
- **Model type:** Causal Language Model
- **Language(s) (NLP):** English
- **License:** Apache 2.0
- **Finetuned from model:** facebook/opt-350m

### Model Sources

- **Repository:** https://github.com/YuvrajSingh-mist/SmolQuant
- **Original OPT models:** https://huggingface.co/facebook/opt-350m

## Uses

### Direct Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load quantized model
model = AutoModelForCausalLM.from_pretrained("YuvrajSingh9886/facebook-opt-350m-8bit-bnb")
tokenizer = AutoTokenizer.from_pretrained("YuvrajSingh9886/facebook-opt-350m-8bit-bnb")

# Generate text
inputs = tokenizer("Hello, my name is", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

### Downstream Use

These quantized models can be used for:
- Text generation
- Language modeling tasks
- Fine-tuning with reduced memory requirements
- Inference on resource-constrained devices

## Bias, Risks, and Limitations

### Recommendations

Users should be aware that these models may:
- Produce biased or inappropriate content
- Have reduced accuracy compared to full-precision models
- Require careful prompt engineering for optimal results

### Known Limitations

- Quantization may introduce slight accuracy degradation
- Memory savings come at the cost of numerical precision
- Some edge cases may show different behavior than full-precision models

## How to Get Started with the Model

Use the code below to get started with the model.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# For inference with automatic quantization
model = AutoModelForCausalLM.from_pretrained(
    "YuvrajSingh9886/facebook-opt-350m-8bit-llm.int8-threshold-8",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("YuvrajSingh9886/facebook-opt-350m-8bit-llm.int8-threshold-8")

# Generate text
text = "The future of AI is"
inputs = tokenizer(text, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=50, do_sample=True, temperature=0.7)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

## Training Details

### Training Data

The models are quantized versions of pre-trained OPT-350M models. No additional training data was used - quantization is a post-training compression technique.

### Training Procedure

#### Preprocessing

- **Quantization Method:** LLM.int8 using BitsAndBytes
- **Threshold Settings:** Various threshold values tested (5.0, 6.0, 8.0)
- **Layer Handling:** Language modeling head (lm_head) kept in FP16 for stability

#### Speeds, Sizes, Times

- **Original Model Size:** ~0.6GB (FP16)
- **Quantized Model Size:** ~0.3GB (8-bit weights + FP16 activations)
- **Memory Reduction:** ~50%
- **Inference Speed:** Comparable to FP16 with reduced memory usage

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data
- **Datasets:** hellaswag, piqa, arc_easy, arc_challenge, openbookqa, winogrande, super-glue-lm-eval-v1
- **Language:** English

#### Metrics
- **Accuracy:** Task-specific accuracy scores
- **Perplexity:** Language modeling perplexity
- **Memory Usage:** Peak memory consumption during inference

*Note: Performance numbers are approximate and may vary based on evaluation setup*

### Results

| Model Variant | Memory (GB) | HellaSwag Acc | Relative Perf |
|---------------|-------------|---------------|---------------|
| OPT-350M (FP16) | 0.6 | Baseline | 100% |
| LLM.int8 (threshold=6.0) | 0.3 | -0.5% | 99.5% |
| LLM.int8 (threshold=8.0) | 0.3 | -0.3% | 99.7% |
| LLM.int8 (threshold=5.0, lm_head=FP16) | 0.3 | -0.2% | 99.8% |

*Note: Performance numbers are approximate and may vary based on evaluation setup*

## Technical Specifications

### Model Architecture and Objective

The models maintain the original OPT architecture:
- **Layers:** 24 transformer layers
- **Hidden Size:** 1024
- **Attention Heads:** 16
- **Feed-forward Size:** 4096
- **Vocabulary Size:** 50,257

### Compute Infrastructure

- **Hardware:** NVIDIA GPUs (A100, V100, T4)
- **Framework:** PyTorch 2.0+, Transformers 4.21+
- **Quantization:** BitsAndBytes 0.37+

## Model Card Contact

- **Name:** YuvrajSingh-mist
- **Email:** [Contact information]
- **GitHub:** https://github.com/YuvrajSingh-mist
- **Wandb:** https://wandb.ai/rentio/SmolQuant/reports/OPT-350M-Quantization-using-LLM-int8---VmlldzoxNDY5ODE2Ng

## Citation

If you use these quantized models in your work, please cite:
```bibtex
@inproceedings{zhang2022opt,
  title={Opt: Open pre-trained transformer language models},
  author={Zhang, Susan and Roller, Stephen and Goyal, Naman and Artetxe, Mikel and Chen, Moya and Chen, Shuohui and Dewan, Christopher and Diab, Mona and Li, Xian and Lin, Xi Victoria and others},
  booktitle={arXiv preprint arXiv:2205.01068},
  year={2022}
}

@article{dettmers2022llm,
  title={LLM. int8 (): 8-bit matrix multiplication for transformers at scale},
  author={Dettmers, Tim and Lewis, Mike and Belkada, Younes and Zettlemoyer, Luke},
  journal={arXiv preprint arXiv:2208.07339},
  year={2022}
}
```

## Model Card Authors

- **Model Card Author:** YuvrajSingh-mist
- **Model Developer:** YuvrajSingh-mist
- **Original OPT Developers:** Meta AI (Facebook)

---

*This model card follows the HuggingFace model card format and provides comprehensive information about the LLM.int8 quantized OPT models.*