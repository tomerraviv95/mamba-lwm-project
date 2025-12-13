"""
Debug utility to check for NaN/Inf values in the Mamba model.
Run this to verify the model produces valid outputs.
"""

import torch
import torch.nn as nn
import mamba_model

def check_nan_inf(tensor, name):
    """Check if tensor contains NaN or Inf values."""
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    if has_nan or has_inf:
        print(f"  ❌ {name}: NaN={has_nan}, Inf={has_inf}")
        print(f"     Min={tensor.min().item():.4f}, Max={tensor.max().item():.4f}, Mean={tensor.mean().item():.4f}")
        return True
    else:
        print(f"  ✓ {name}: OK (min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, mean={tensor.mean().item():.4f})")
        return False

def debug_forward_pass():
    """Test forward pass and check for numerical issues."""
    print("\n" + "=" * 80)
    print("DEBUG: Testing Mamba Model for NaN/Inf Issues")
    print("=" * 80)

    # Model configuration
    ELEMENT_LENGTH = 32
    D_MODEL = 128
    N_LAYERS = 12
    MAX_LEN = 513
    D_STATE = 16
    D_CONV = 4
    EXPAND = 2
    DROPOUT = 0.0  # Disable dropout for debugging

    print("\n[1/5] Initializing model...")
    model = mamba_model.lwm_mamba(
        element_length=ELEMENT_LENGTH,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        max_len=MAX_LEN,
        d_state=D_STATE,
        d_conv=D_CONV,
        expand=EXPAND,
        dropout=DROPOUT,
    )
    model.to("cuda")
    model.eval()  # Use eval mode to disable dropout

    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create test input
    batch_size = 2
    seq_len = 33
    print(f"\n[2/5] Creating test input: ({batch_size}, {seq_len}, {ELEMENT_LENGTH})")

    # Use smaller random values for input
    input_ids = torch.randn(batch_size, seq_len, ELEMENT_LENGTH) * 0.1
    input_ids = input_ids.to("cuda")
    check_nan_inf(input_ids, "Input")

    # Test forward pass without masking
    print(f"\n[3/5] Testing forward pass (no masking)...")
    with torch.no_grad():
        try:
            output = model(input_ids)
            has_issue = check_nan_inf(output, "Output")
            if has_issue:
                print("\n❌ NaN/Inf detected in output!")
            else:
                print("\n✓ Forward pass successful (no masking)")
        except Exception as e:
            print(f"\n❌ Error during forward pass: {e}")
            return False

    # Test forward pass with masking
    print(f"\n[4/5] Testing forward pass (with masking)...")
    num_masked = int(seq_len * 0.15)
    masked_pos = torch.randint(0, seq_len, (batch_size, num_masked))
    masked_pos = masked_pos.to("cuda")
    
    with torch.no_grad():
        try:
            logits_lm, output = model(input_ids, masked_pos)
            has_issue_logits = check_nan_inf(logits_lm, "Masked logits")
            has_issue_output = check_nan_inf(output, "Output")

            if has_issue_logits or has_issue_output:
                print("\n❌ NaN/Inf detected with masking!")
            else:
                print("\n✓ Forward pass successful (with masking)")
        except Exception as e:
            print(f"\n❌ Error during forward pass with masking: {e}")
            return False

    # Test with gradient computation
    print(f"\n[5/5] Testing backward pass...")
    model.train()
    input_ids.requires_grad = True

    try:
        logits_lm, output = model(input_ids, masked_pos)

        # Create dummy target and compute loss
        target = torch.randn_like(logits_lm)
        loss = nn.MSELoss()(logits_lm, target)

        print(f"  Loss value: {loss.item():.6f}")
        check_nan_inf(loss, "Loss")

        # Backward pass
        loss.backward()

        # Check gradients
        grad_issues = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"  ❌ {name}: gradient has NaN/Inf")
                    grad_issues = True

        if not grad_issues:
            print("  ✓ All gradients are valid")
        else:
            print("\n❌ Gradient issues detected!")
            return False

    except Exception as e:
        print(f"\n❌ Error during backward pass: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED - Model is numerically stable!")
    print("=" * 80 + "\n")
    return True

if __name__ == "__main__":
    success = debug_forward_pass()
    if not success:
        print("\n⚠ Issues detected. Check the output above for details.")
        print("Common fixes:")
        print("  1. Reduce learning rate (try 1e-4 instead of 1e-3)")
        print("  2. Use gradient clipping (torch.nn.utils.clip_grad_norm_)")
        print("  3. Check input data normalization")
        print("  4. Increase warmup steps")
    exit(0 if success else 1)
