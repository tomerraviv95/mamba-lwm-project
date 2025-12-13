"""
Detailed parameter analysis for Transformer vs Mamba architectures.
This script breaks down parameters by component to understand the differences.
"""

import torch
import pretrained_model
import mamba_model

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def analyze_model_parameters(model, model_name):
    """Detailed parameter breakdown by component."""
    print(f"\n{'='*80}")
    print(f"{model_name} - Parameter Breakdown")
    print(f"{'='*80}")

    total = 0
    for name, module in model.named_children():
        params = count_parameters(module)
        total += params
        print(f"{name:20s}: {params:>12,} parameters")

        # If it's a ModuleList (like layers), show per-layer breakdown
        if isinstance(module, torch.nn.ModuleList):
            if len(module) > 0:
                single_layer_params = count_parameters(module[0])
                print(f"  → Per layer:      {single_layer_params:>12,} parameters")
                print(f"  → Num layers:     {len(module):>12,}")

    print(f"{'-'*80}")
    print(f"{'TOTAL':20s}: {total:>12,} parameters")
    print(f"{'='*80}")

    return total

def detailed_layer_breakdown(model, model_name):
    """Show parameter breakdown within a single layer."""
    print(f"\n{'='*80}")
    print(f"{model_name} - Single Layer Breakdown")
    print(f"{'='*80}")

    if hasattr(model, 'layers') and len(model.layers) > 0:
        layer = model.layers[0]

        for name, param in layer.named_parameters():
            print(f"{name:50s}: {param.numel():>10,} ({list(param.shape)})")

        print(f"{'-'*80}")
        print(f"{'Layer Total':50s}: {count_parameters(layer):>10,}")
    print(f"{'='*80}")

def main():
    # Hyperparameters
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

    print("\n" + "="*80)
    print("PARAMETER ANALYSIS: TRANSFORMER vs MAMBA")
    print("="*80)
    print(f"\nShared hyperparameters:")
    print(f"  element_length = {ELEMENT_LENGTH}")
    print(f"  d_model = {D_MODEL}")
    print(f"  n_layers = {N_LAYERS}")
    print(f"  max_len = {MAX_LEN}")
    print(f"  dropout = {DROPOUT}")

    print(f"\nTransformer-specific:")
    print(f"  n_heads = {N_HEADS}")

    print(f"\nMamba-specific:")
    print(f"  d_state = {D_STATE}")
    print(f"  d_conv = {D_CONV}")
    print(f"  expand = {EXPAND}")

    # Initialize models
    print("\n[1/4] Initializing Transformer model...")
    transformer_model = pretrained_model.lwm(
        element_length=ELEMENT_LENGTH,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        max_len=MAX_LEN,
        n_heads=N_HEADS,
        dropout=DROPOUT,
    )

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

    # Analyze parameters
    print("\n[3/4] Analyzing parameter distribution...")
    transformer_total = analyze_model_parameters(transformer_model, "TRANSFORMER")
    mamba_total = analyze_model_parameters(mamba_mdl, "MAMBA")

    # Detailed layer breakdown
    print("\n[4/4] Detailed layer analysis...")
    detailed_layer_breakdown(transformer_model, "TRANSFORMER")
    detailed_layer_breakdown(mamba_mdl, "MAMBA")

    # Comparison
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"Transformer total: {transformer_total:>12,} parameters")
    print(f"Mamba total:       {mamba_total:>12,} parameters")
    print(f"Difference:        {mamba_total - transformer_total:>+12,} parameters ({(mamba_total/transformer_total - 1)*100:+.1f}%)")
    print(f"{'='*80}")

    # Calculate Mamba parameter formulas
    d_inner = int(EXPAND * D_MODEL)
    print(f"\nMamba Parameter Calculations:")
    print(f"  d_inner = expand × d_model = {EXPAND} × {D_MODEL} = {d_inner}")
    print(f"\n  Per MambaBlock parameters:")
    print(f"    in_proj:    d_model × (2 × d_inner) = {D_MODEL} × {2*d_inner} = {D_MODEL * 2 * d_inner:,}")
    print(f"    conv1d:     d_inner × d_conv = {d_inner} × {D_CONV} = {d_inner * D_CONV:,}")
    print(f"    x_proj:     d_inner × (2 × d_state) = {d_inner} × {2*D_STATE} = {d_inner * 2 * D_STATE:,}")
    print(f"    dt_proj:    d_inner × d_inner = {d_inner} × {d_inner} = {d_inner * d_inner:,}")
    print(f"    A_log:      d_inner × d_state = {d_inner} × {D_STATE} = {d_inner * D_STATE:,}")
    print(f"    D:          d_inner = {d_inner:,}")
    print(f"    out_proj:   d_inner × d_model = {d_inner} × {D_MODEL} = {d_inner * D_MODEL:,}")
    print(f"    LayerNorm:  2 × d_model = 2 × {D_MODEL} = {2 * D_MODEL:,}")

    print(f"\n  Per MambaLayer (MambaBlock + FFN):")
    print(f"    FFN:        d_model × d_ff + d_ff × d_model = {D_MODEL} × {D_MODEL*4} + {D_MODEL*4} × {D_MODEL} = {D_MODEL * D_MODEL * 4 * 2:,}")

    print(f"\n{'='*80}")
    print("RECOMMENDATIONS TO REDUCE MAMBA PARAMETERS")
    print(f"{'='*80}")

    # Calculate different configurations
    configs = [
        ("Current", EXPAND, D_STATE, D_CONV),
        ("Reduce expand to 1.5", 1.5, D_STATE, D_CONV),
        ("Reduce d_state to 12", EXPAND, 12, D_CONV),
        ("Reduce d_state to 8", EXPAND, 8, D_CONV),
        ("Reduce expand to 1.5 + d_state to 12", 1.5, 12, D_CONV),
    ]

    print(f"\n{'Configuration':45s} {'Expand':>8s} {'D_State':>8s} {'Est. Params':>15s}")
    print(f"{'-'*80}")

    for config_name, expand, d_state, d_conv in configs:
        # Rough estimate based on dominant terms
        d_inner_est = int(expand * D_MODEL)
        # Main contributors per layer:
        # in_proj: D_MODEL * 2 * d_inner
        # dt_proj: d_inner * d_inner (biggest)
        # x_proj: d_inner * 2 * d_state
        # out_proj: d_inner * D_MODEL
        # A_log: d_inner * d_state
        # FFN: 2 * D_MODEL * 4 * D_MODEL

        mamba_block_params = (
            D_MODEL * 2 * d_inner_est +  # in_proj
            d_inner_est * d_conv +  # conv1d
            d_inner_est * 2 * d_state +  # x_proj
            d_inner_est * d_inner_est +  # dt_proj (biggest!)
            d_inner_est * d_state +  # A_log
            d_inner_est +  # D
            d_inner_est * D_MODEL +  # out_proj
            2 * D_MODEL  # LayerNorms
        )

        ffn_params = 2 * D_MODEL * 4 * D_MODEL + D_MODEL  # FFN + LayerNorm

        layer_params = mamba_block_params + ffn_params
        total_est = (
            D_MODEL * ELEMENT_LENGTH + MAX_LEN * D_MODEL +  # embedding
            layer_params * N_LAYERS +  # layers
            D_MODEL * ELEMENT_LENGTH  # decoder
        )

        print(f"{config_name:45s} {expand:>8.1f} {d_state:>8d} {total_est:>15,}")

    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
