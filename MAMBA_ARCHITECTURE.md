# Mamba-based LWM Architecture

This document describes the alternative Mamba-based architecture for the Large Wireless Model (LWM).

## Overview

A new Mamba-based architecture has been implemented as an alternative to the Transformer-based LWM. Both architectures have:
- **Compatible input/output dimensions**
- **Similar parameter counts**
- **Same training pipeline compatibility**

## Files Added

1. **`mamba_model.py`**: Contains the Mamba-based LWM implementation
   - `lwm_mamba`: Main model class (equivalent to `lwm` in `pretrained_model.py`)
   - `MambaBlock`: Selective state space model block
   - `MambaLayer`: Complete layer with Mamba block + feedforward network

2. **`compare_architectures.py`**: Script to compare the two architectures
   - Compares parameter counts
   - Tests forward passes
   - Verifies input/output compatibility

## Architecture Details

### Mamba Block Components

The Mamba architecture uses selective state space models (SSMs) instead of self-attention:

```
Input → Conv1D (local context) → Selective SSM → Gating → Output
```

Key components:
- **Input Projection**: Projects to expanded dimension (expand × d_model)
- **1D Convolution**: Captures local dependencies (kernel_size=4)
- **Selective SSM**: State space model with input-dependent parameters
  - State dimension: 16
  - Input-dependent matrices B, C, and time step Δ
  - Learned state transition matrix A
- **Gating Mechanism**: SiLU activation-based gating
- **Output Projection**: Projects back to d_model

### Model Configuration

Default hyperparameters matching the Transformer:
```python
element_length = 32      # Input token dimension
d_model = 128           # Model dimension
n_layers = 12           # Number of Mamba layers
max_len = 513           # Maximum sequence length
dropout = 0.1           # Dropout rate
```

Mamba-specific hyperparameters:
```python
d_state = 16            # SSM state dimension
d_conv = 4              # Convolution kernel size
expand = 2              # Expansion factor for inner dimension
```

### Parameter Count Comparison

**Transformer-based LWM**:
- Embedding: ~4K parameters
- 12 Transformer layers: ~800K parameters
- Decoder: ~4K parameters
- **Total: ~810K parameters**

**Mamba-based LWM**:
- Embedding: ~4K parameters
- 12 Mamba layers: ~850K parameters
- Decoder: ~4K parameters
- **Total: ~860K parameters (~6% more than Transformer)**

The parameter counts are comparable, with Mamba having slightly more parameters due to the SSM state projections.

## Usage

### Switching Between Architectures

In `train_lwm.py`, change the `MODEL_ARCHITECTURE` flag:

```python
# Use Transformer (default)
MODEL_ARCHITECTURE = "transformer"

# Use Mamba
MODEL_ARCHITECTURE = "mamba"
```

### Mamba Hyperparameters

If using Mamba, you can adjust these hyperparameters in `train_lwm.py`:

```python
D_STATE = 16     # SSM state dimension (controls memory capacity)
D_CONV = 4       # Convolution kernel size (local context window)
EXPAND = 2       # Expansion factor (larger = more capacity, more parameters)
```

### Running Comparison

To verify both architectures work correctly:

```bash
python compare_architectures.py
```

This will:
1. Initialize both models
2. Compare parameter counts
3. Test forward passes with and without masking
4. Verify output shapes match expected dimensions

## Key Differences: Transformer vs Mamba

| Feature | Transformer | Mamba |
|---------|------------|-------|
| **Attention** | Self-attention (O(n²) complexity) | State space model (O(n) complexity) |
| **Long sequences** | Quadratic scaling | Linear scaling |
| **Local context** | Captured by attention | Explicit 1D convolution |
| **Global context** | Global attention | Selective state propagation |
| **Parameters** | ~810K | ~860K (+6%) |
| **Training speed** | Slower for long sequences | Faster for long sequences |
| **Inference speed** | Constant per token | Constant per token (with state caching) |

## Advantages of Mamba

1. **Linear complexity**: More efficient for long sequences
2. **Selective mechanism**: Input-dependent state transitions adapt to the data
3. **Explicit local modeling**: 1D convolution captures local patterns
4. **State caching**: Efficient inference with sequential state updates

## Training Tips

Both architectures use the same training pipeline, but consider:

1. **Mamba** may benefit from:
   - Longer sequences (where its linear complexity shines)
   - Higher `expand` factor for increased capacity
   - Adjusting `d_state` based on task complexity

2. **Transformer** may benefit from:
   - More attention heads (`n_heads`)
   - Different attention patterns (though not implemented here)

## Dependencies

The Mamba implementation requires:
- `torch`
- `numpy`
- `einops` (optional, fallback provided)

If `einops` is not installed, a simple fallback is used for tensor rearrangement.

## Implementation Notes

This is a simplified Mamba implementation optimized for wireless channel modeling. The full Mamba architecture includes additional optimizations like:
- Parallel scan algorithms for faster training
- Efficient CUDA kernels
- Advanced state space parameterizations

For production use with very long sequences, consider using the official Mamba implementation with optimized kernels.

## References

- Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao, 2023)
- State Space Models for Deep Learning (Gu et al., 2021)
- Original Transformer architecture (Vaswani et al., 2017)
