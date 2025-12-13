"""
Test script to verify the mamba-ssm integration works correctly.
"""
import torch
from mamba_model import lwm_mamba, MAMBA_SSM_AVAILABLE

def test_model_forward():
    """Test that the model forward pass works with the official implementation."""
    print(f"MAMBA_SSM_AVAILABLE: {MAMBA_SSM_AVAILABLE}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create a small model for testing
    model = lwm_mamba(
        element_length=32,
        d_model=128,
        n_layers=2,  # Use fewer layers for quick testing
        max_len=513,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1
    ).to(device)

    print(f"Model created successfully")
    print(f"Using official mamba-ssm: {model.layers[0].mamba.use_official if hasattr(model.layers[0].mamba, 'use_official') else 'N/A'}")

    # Create dummy input
    batch_size = 2
    seq_len = 64
    element_length = 32

    input_ids = torch.randn(batch_size, seq_len, element_length).to(device)

    # Test forward pass without masked_pos
    print("\nTesting forward pass without masked_pos...")
    with torch.no_grad():
        output = model(input_ids)

    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, seq_len, 128), f"Expected shape {(batch_size, seq_len, 128)}, got {output.shape}"
    print("✓ Forward pass without masked_pos works!")

    # Test forward pass with masked_pos
    print("\nTesting forward pass with masked_pos...")
    num_masked = 10
    masked_pos = torch.randint(0, seq_len, (batch_size, num_masked)).to(device)

    with torch.no_grad():
        logits_lm, output = model(input_ids, masked_pos=masked_pos)

    print(f"Logits shape: {logits_lm.shape}")
    print(f"Output shape: {output.shape}")
    assert logits_lm.shape == (batch_size, num_masked, 32), f"Expected logits shape {(batch_size, num_masked, 32)}, got {logits_lm.shape}"
    print("✓ Forward pass with masked_pos works!")

    # Test gradient computation
    print("\nTesting gradient computation...")
    model.train()
    output = model(input_ids)
    loss = output.sum()
    loss.backward()
    print("✓ Gradient computation works!")

    print("\n" + "="*50)
    print("All tests passed! ✓")
    print("="*50)

if __name__ == "__main__":
    test_model_forward()
