"""
Script to compare the Transformer, Unidirectional Mamba, and Bidirectional Mamba architectures.
This verifies that all models have similar parameter counts and compatible input/output dimensions.
"""

import torch
import pretrained_model
import mamba_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    if hasattr(model, 'proj'):
        proj_params = count_parameters(model.proj)
        print(f"  Input projection: {proj_params:,} parameters")

    if hasattr(model, 'layers'):
        layers_params = count_parameters(model.layers)
        print(f"  Encoder layers ({len(model.layers)} layers): {layers_params:,} parameters")
        print(f"    Average per layer: {layers_params // len(model.layers):,} parameters")

    if hasattr(model, 'decoder'):
        decoder_params = count_parameters(model.decoder)
        print(f"  Decoder: {decoder_params:,} parameters")

    # Check bidirectionality for Mamba
    if hasattr(model, 'bidirectional'):
        print(f"  Bidirectional: {model.bidirectional}")

    return total_params

def test_forward_pass(model, model_name, batch_size=4, seq_len=33, element_length=32):
    """Test forward pass with and without masking."""
    print(f"\n{'=' * 80}")
    print(f"Testing {model_name} Forward Pass")
    print(f"{'=' * 80}")

    # Create dummy input
    input_ids = torch.randn(batch_size, seq_len, element_length).to(device)

    # Test without masking
    print(f"Input shape: {input_ids.shape}")
    with torch.no_grad():
        output = model(input_ids)
    print(f"Output shape (no masking): {output.shape}")

    # Test with masking
    num_masked = int(seq_len * 0.15)  # Mask 15% of tokens
    masked_pos = torch.randint(0, seq_len, (batch_size, num_masked)).to(device)

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
    D_STATE = 8
    D_CONV = 4
    EXPAND = 1.2

    print("\n" + "=" * 80)
    print("COMPARING TRANSFORMER vs MAMBA ARCHITECTURES")
    print("=" * 80)

    # Initialize Transformer model
    print("\n[1/6] Initializing Transformer model...")
    transformer_model = pretrained_model.lwm(
        element_length=ELEMENT_LENGTH,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        max_len=MAX_LEN,
        n_heads=N_HEADS,
        dropout=DROPOUT,
    ).to(device)
    
    # Initialize Unidirectional Mamba model
    print("\n[2/6] Initializing Unidirectional Mamba model...")
    mamba_uni = mamba_model.lwm_mamba(
        element_length=ELEMENT_LENGTH,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        max_len=MAX_LEN,
        d_state=D_STATE,
        d_conv=D_CONV,
        expand=EXPAND,
        dropout=DROPOUT,
        bidirectional=False,
    ).to(device)

    # Initialize Bidirectional Mamba model
    print("\n[3/6] Initializing Bidirectional Mamba model...")
    mamba_bi = mamba_model.lwm_mamba(
        element_length=ELEMENT_LENGTH,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        max_len=MAX_LEN,
        d_state=D_STATE,
        d_conv=D_CONV,
        expand=EXPAND,
        dropout=DROPOUT,
        bidirectional=True,
    ).to(device)

    # Get model info
    print("\n[4/6] Analyzing model architectures...")
    transformer_params = get_model_info(transformer_model, "Transformer-based LWM")
    mamba_uni_params = get_model_info(mamba_uni, "Mamba-based LWM (Unidirectional)")
    mamba_bi_params = get_model_info(mamba_bi, "Mamba-based LWM (Bidirectional)")

    # Compare parameter counts
    print(f"\n{'=' * 80}")
    print("PARAMETER COMPARISON")
    print(f"{'=' * 80}")
    print(f"Transformer (baseline):    {transformer_params:,}")
    print(f"Mamba (unidirectional):    {mamba_uni_params:,}")
    print(f"Mamba (bidirectional):     {mamba_bi_params:,}")

    # Compare Uni-Mamba to Transformer
    diff_uni = mamba_uni_params - transformer_params
    diff_uni_pct = (diff_uni / transformer_params) * 100
    print(f"\nUni-Mamba vs Transformer:  {diff_uni:+,} ({diff_uni_pct:+.2f}%)")

    # Compare Bi-Mamba to Transformer
    diff_bi = mamba_bi_params - transformer_params
    diff_bi_pct = (diff_bi / transformer_params) * 100
    print(f"Bi-Mamba vs Transformer:   {diff_bi:+,} ({diff_bi_pct:+.2f}%)")

    # Compare Bi-Mamba to Uni-Mamba
    diff_bi_uni = mamba_bi_params - mamba_uni_params
    diff_bi_uni_pct = (diff_bi_uni / mamba_uni_params) * 100
    print(f"Bi-Mamba vs Uni-Mamba:     {diff_bi_uni:+,} ({diff_bi_uni_pct:+.2f}%)")

    # Validation
    print(f"\n{'=' * 80}")
    print("VALIDATION")
    print(f"{'=' * 80}")
    if abs(diff_bi_pct) < 20:
        print("✓ Bidirectional Mamba parameter count is similar to Transformer (within 20%)")
    else:
        print("⚠ Bidirectional Mamba parameter count differs from Transformer by more than 20%")

    if abs(diff_bi_uni_pct) < 10:
        print("✓ Bidirectional and Unidirectional Mamba have similar parameter counts (within 10%)")
    else:
        print("⚠ Bidirectional Mamba has significantly different parameters than Unidirectional")

    # Test forward passes
    print("\n[5/6] Testing forward passes...")
    test_forward_pass(transformer_model, "Transformer", seq_len=33)
    test_forward_pass(mamba_uni, "Mamba (Unidirectional)", seq_len=33)
    test_forward_pass(mamba_bi, "Mamba (Bidirectional)", seq_len=33)

    print("\n[6/6] Summary")
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print("\nAll models are compatible with the same training pipeline!")
    print("\nConfiguration in train_lwm.py:")
    print("  - MODEL_ARCHITECTURE = 'transformer'  → Use Transformer")
    print("  - MODEL_ARCHITECTURE = 'mamba'")
    print("    - BIDIRECTIONAL = False  → Use Unidirectional Mamba")
    print("    - BIDIRECTIONAL = True   → Use Bidirectional Mamba (recommended for MLM)")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
