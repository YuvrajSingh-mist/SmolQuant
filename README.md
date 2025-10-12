# SmolQuant 🔢

A lightweight PyTorch implementation of 8-bit weight quantization for neural networks, focusing on Linear layers with W8A16 (8-bit weights, 16-bit activations) quantization.

## Overview

SmolQuant provides an efficient way to quantize PyTorch models by replacing standard `nn.Linear` layers with custom quantized implementations. This reduces memory usage and can improve inference speed while maintaining model accuracy.

## Features

- **W8A16 Quantization**: 8-bit weights with 16-bit activations
- **Per-channel Quantization**: Individual scaling factors for each output channel
- **Seamless Integration**: Drop-in replacement for `nn.Linear` layers
- **CUDA Support**: Automatic device handling for GPU acceleration
- **Bias Preservation**: Maintains original bias values from pre-trained models
- **Transformer Support**: Tested with HuggingFace transformers

## Installation

```bash
git clone https://github.com/YuvrajSingh-mist/SmolQuant.git
cd SmolQuant
pip install torch transformers matplotlib
```

## Quick Start

### Basic Usage

```python
import torch
import torch.nn as nn
from your_quantization_module import W8A16LinearLayer, replace_linear_with_target_and_quantize

# Create a simple model
model = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

# Quantize all linear layers
replace_linear_with_target_and_quantize(model, W8A16LinearLayer, [])
print("Model quantized successfully!")
```

### Quantizing Transformer Models

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load a pre-trained model
model = AutoModelForCausalLM.from_pretrained(
    "Salesforce/codegen-350M-mono",
    torch_dtype=torch.bfloat16
)

# Quantize all linear layers except the language model head
replace_linear_with_target_and_quantize(model, W8A16LinearLayer, ["lm_head"])

# Use the quantized model
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
```

## How It Works

### Quantization Process

1. **Linear Quantization**: Maps float32 weights to int8 range [-128, 127]
   ```
   quantized = round(weight / scale + zero_point)
   ```

2. **Per-channel Scaling**: Each output channel gets its own scale and zero-point
   ```
   scale = (max_val - min_val) / (127 - (-128))
   zero_point = -128 - min_val / scale
   ```

3. **Dequantization**: Converts back to float for computation
   ```
   dequantized = scale * (quantized - zero_point)
   ```

### Architecture

```
W8A16LinearLayer
├── int8_weights: Quantized weights stored as int8
├── scales: Per-channel scaling factors
├── zero_point: Per-channel zero points
├── bias: Original bias values (if present)
└── forward(): Dequantize → MatMul → Add bias
```

## Performance

| Model | Original Size | Quantized Size | Memory Reduction | Accuracy Loss |
|-------|---------------|----------------|------------------|---------------|
| CodeGen-350M | 1.4GB | ~700MB | ~50% | <2% |
| GPT-2 Small | 500MB | ~250MB | ~50% | <1% |

*Results may vary based on model architecture and quantization settings*

## API Reference

### Core Functions

#### `linear_quantize(tensor, dtype=torch.int8)`
Quantizes a single tensor to the specified integer type.

**Parameters:**
- `tensor`: Input float tensor to quantize
- `dtype`: Target quantization dtype (default: torch.int8)

**Returns:**
- `scale`: Scaling factor for dequantization
- `zero_point`: Zero point offset
- `quantized_tensor`: Quantized tensor

#### `channel_linear_quantize(tensor, dim=0, dtype=torch.int8)`
Performs per-channel quantization along the specified dimension.

**Parameters:**
- `tensor`: Input tensor
- `dim`: Dimension along which to quantize (0 for output channels)
- `dtype`: Target quantization dtype

#### `W8A16LinearLayer(in_features, out_features, dtype=torch.float32, bias=None)`
Custom quantized linear layer implementation.

**Parameters:**
- `in_features`: Input dimension
- `out_features`: Output dimension  
- `dtype`: Computation dtype
- `bias`: Bias tensor (None for no bias)

#### `replace_linear_with_target_and_quantize(module, target_class, module_name_to_exclude)`
Recursively replaces Linear layers in a model with quantized versions.

**Parameters:**
- `module`: PyTorch module to quantize
- `target_class`: Quantized layer class (e.g., W8A16LinearLayer)
- `module_name_to_exclude`: List of layer names to skip

## Examples

### Measuring Quantization Error

```python
# Test quantization accuracy
quantized_layer = W8A16LinearLayer(128, 64)
original_weights = torch.randn(64, 128)

# Quantize and dequantize
scales, zero_pts, quant = quantized_layer.quantize(original_weights)
reconstructed = quantized_layer.linear_dequantize(scales, zero_pts, quant)

# Calculate MSE
mse = (original_weights - reconstructed).pow(2).mean()
print(f"Reconstruction MSE: {mse:.6f}")
```

### Custom Quantization Settings

```python
# Skip specific layers during quantization
exclude_layers = ["classifier", "lm_head", "embed_tokens"]
replace_linear_with_target_and_quantize(model, W8A16LinearLayer, exclude_layers)
```

### Development Setup

```bash
git clone https://github.com/YuvrajSingh-mist/SmolQuant.git
cd SmolQuant
pip install -r requirements.txt
```

### Running Tests

```bash
python -m pytest tests/
# Or run specific notebooks
jupyter notebook "Linear Quantization/8_bit_quant.ipynb"
```

## Citation

If you use SmolQuant in your research, please cite:

```bibtex
@software{smolquant2024,
  title={SmolQuant: Lightweight Neural Network Quantization},
  author={YuvrajSingh-mist},
  year={2024},
  url={https://github.com/YuvrajSingh-mist/SmolQuant}
}
```

## Acknowledgments

- PyTorch team for the excellent framework
- HuggingFace for transformer implementations
- Quantization research community for foundational work

---

**Made with ❤️ by [YuvrajSingh-mist](https://github.com/YuvrajSingh-mist)**