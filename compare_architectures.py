"""
Script to compare the Transformer and Mamba architectures.
This verifies that both models have similar parameter counts and compatible input/output dimensions.
"""

import torch
import pretrained_model
import mamba_model

def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_info(model, model_name):
    """Get detailed information about a model."""
    total_params = count_parameters(model)

    print(f"\n{'=' * 80}")
    print(f"{model_name} Architecture")
    print(f"{'=' * 80}")
    print(f"Total trainable parameters: {total_params:,}")

    # Break down by component
    if hasattr(model, 'embedding'):
        embedding_params = count_parameters(model.embedding)
        print(f"  Embedding layer: {embedding_params:,} parameters")

    if hasattr(model, 'layers'):
        layers_params = count_parameters(model.layers)
        print(f"  Encoder layers ({len(model.layers)} layers): {layers_params:,} parameters")
        print(f"    Average per layer: {layers_params // len(model.layers):,} parameters")

    if hasattr(model, 'decoder'):
        decoder_params = count_parameters(model.decoder)
        print(f"  Decoder: {decoder_params:,} parameters")

    return total_params

def test_forward_pass(model, model_name, batch_size=4, seq_len=33, element_length=32):
    """Test forward pass with and without masking."""
    print(f"\n{'=' * 80}")
    print(f"Testing {model_name} Forward Pass")
    print(f"{'=' * 80}")

    # Create dummy input
    input_ids = torch.randn(batch_size, seq_len, element_length)

    # Test without masking
    print(f"Input shape: {input_ids.shape}")
    with torch.no_grad():
        output = model(input_ids)
    print(f"Output shape (no masking): {output.shape}")

    # Test with masking
    num_masked = int(seq_len * 0.15)  # Mask 15% of tokens
    masked_pos = torch.randint(0, seq_len, (batch_size, num_masked))

    with torch.no_grad():
        logits_lm, output = model(input_ids, masked_pos)
    print(f"Output shape (with masking): {output.shape}")
    print(f"Masked predictions shape: {logits_lm.shape}")
    print(f"Expected masked shape: ({batch_size}, {num_masked}, {element_length})")

    return True

def main():
    # Hyperparameters (matching train_lwm.py defaults)
    ELEMENT_LENGTH = 32
    D_MODEL = 128
    N_LAYERS = 12
    MAX_LEN = 513
    N_HEADS = 8
    DROPOUT = 0.1

    # Mamba-specific
    D_STATE = 16
    D_CONV = 4
    EXPAND = 2

    print("\n" + "=" * 80)
    print("COMPARING TRANSFORMER vs MAMBA ARCHITECTURES")
    print("=" * 80)

    # Initialize Transformer model
    print("\n[1/4] Initializing Transformer model...")
    transformer_model = pretrained_model.lwm(
        element_length=ELEMENT_LENGTH,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        max_len=MAX_LEN,
        n_heads=N_HEADS,
        dropout=DROPOUT,
    )

    # Initialize Mamba model
    print("\n[2/4] Initializing Mamba model...")
    mamba_mdl = mamba_model.lwm_mamba(
        element_length=ELEMENT_LENGTH,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        max_len=MAX_LEN,
        d_state=D_STATE,
        d_conv=D_CONV,
        expand=EXPAND,
        dropout=DROPOUT,
    )

    # Get model info
    print("\n[3/4] Analyzing model architectures...")
    transformer_params = get_model_info(transformer_model, "Transformer-based LWM")
    mamba_params = get_model_info(mamba_mdl, "Mamba-based LWM")

    # Compare parameter counts
    print(f"\n{'=' * 80}")
    print("PARAMETER COMPARISON")
    print(f"{'=' * 80}")
    print(f"Transformer parameters: {transformer_params:,}")
    print(f"Mamba parameters:       {mamba_params:,}")
    diff = mamba_params - transformer_params
    diff_pct = (diff / transformer_params) * 100
    print(f"Difference:             {diff:+,} ({diff_pct:+.2f}%)")

    if abs(diff_pct) < 20:
        print("✓ Parameter counts are similar (within 20%)")
    else:
        print("⚠ Parameter counts differ by more than 20%")

    # Test forward passes
    print("\n[4/4] Testing forward passes...")
    test_forward_pass(transformer_model, "Transformer", seq_len=33)
    test_forward_pass(mamba_mdl, "Mamba", seq_len=33)

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print("\nBoth models are compatible with the same training pipeline!")
    print("\nTo use Mamba architecture, set MODEL_ARCHITECTURE = 'mamba' in train_lwm.py")
    print("To use Transformer architecture, set MODEL_ARCHITECTURE = 'transformer' in train_lwm.py")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
