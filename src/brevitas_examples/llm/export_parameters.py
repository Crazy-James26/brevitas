import os
import re
from typing import Optional, Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import brevitas.nn as qnn


def export_parameters(
    model: torch.nn.Module,
    decoder_layer_num: int = 16,    # rows for *_s_sum.h and attention headers
    head_num: int = 32,             # Q/A heads
    kv_head_num: int = 8,           # K/V heads (grouped KV)
    hidden_dim: int = 2048,         # used to infer Dq when head_dim is not exposed
    kv_hidden_dim: int = 512,       # used to infer Dkv when head_dim is not exposed
    bin_dir: str = "weights_bin",
    hdr_dir: str = "weights_headers",
) -> None:
    """
    Unified exporter:
      1) For every qnn.QuantLinear:
         - <leaf>_Lxx.bin (int8) with qw_quant = round(clamp(qw.value / qw.scale, -8..7)),
           layout row-major [out_features, in_features].
         - w_<leaf>_s_sum.h with:
             static const float w_<leaf>_s[DECODER_LAYER_NUM][OUT_FEATURES];
             static const float w_<leaf>_sum[DECODER_LAYER_NUM][OUT_FEATURES];
           where w_*_s is per-output-channel scale (float),
           and w_*_sum is sum over input dim of ORIGINAL float weights (float).
      2) Attention scales:
         - Q_s.h, A_s.h use shape [DECODER_LAYER_NUM][head_num].
         - K_s.h, V_s.h use shape [DECODER_LAYER_NUM][kv_head_num].
      3) RMSNorm gamma (weight):
         - w_rmsnorm.h with:
             static const float RMSNorm_weight[2*DECODER_LAYER_NUM+1][HIDDEN_DIM];
           Row order: for each layer L, rows 2*L (input_layernorm) and 2*L+1 (post_attention_layernorm),
           final row index 2*DECODER_LAYER_NUM is model.norm.
      4) nn.Embedding weights:
         - <sanitized_name>_fp32.bin in row-major [num_embeddings, embedding_dim].

    If actual model sizes differ, arrays are padded/truncated to the provided fixed dimensions.
    """

    # ---------------- Helpers ----------------
    def ensure_dir(d: str):
        os.makedirs(d, exist_ok=True)

    def sanitize_name(n: str) -> str:
        return re.sub(r"[^A-Za-z0-9_]", "_", n)

    def get_layer_idx(mod_name: str) -> Optional[int]:
        m = re.search(r"\blayers\.(\d+)\b", mod_name)
        return int(m.group(1)) if m else None

    def get_leaf_name(mod_name: str) -> str:
        if mod_name.split(".")[-1] == "layer":
            return mod_name.split(".")[-2]
        return mod_name.split(".")[-1]

    def to_per_out_channel_scale(qw_scale: torch.Tensor, out_features: int) -> torch.Tensor:
        """
        Normalize scales to shape [out_features], handling scalar and broadcastable shapes.
        """
        s = qw_scale
        if s.numel() == 1:
            return s.view(1).repeat(out_features)
        if s.numel() == out_features:
            return s.view(out_features)
        # Fallback: flatten and slice
        return s.reshape(-1)[:out_features]

    def header_guard(name: str) -> str:
        return re.sub(r"[^A-Z0-9_]", "_", name.upper()) + "_H_"

    def c_array_2d_float(name: str, arr_2d: np.ndarray) -> str:
        rows, cols = arr_2d.shape
        lines = [f"static const float {name}[{rows}][{cols}] = {{"]
        for r in range(rows):
            row_vals = ", ".join(f"{float(v):.9g}" for v in arr_2d[r])
            lines.append(f"  {{ {row_vals} }},")
        lines.append("};")
        return "\n".join(lines)

    def per_head_from_scale_array(s_arr: np.ndarray, target_H: int) -> np.ndarray:
        """
        Reduce an arbitrary-shaped scale array to a length target_H vector (float32).
        Preference: use an axis of size target_H; else use the largest axis >1; else scalar.
        Then pad/truncate to length target_H.
        """
        s = np.array(s_arr)
        if s.ndim == 0:
            return np.full((target_H,), float(s), dtype=np.float32)
        if target_H in s.shape:
            axis = list(s.shape).index(target_H)
            moved = np.moveaxis(s, axis, 0)
            raw = moved.reshape(target_H, -1).mean(axis=1).astype(np.float32)
        else:
            axes = [dim for dim in s.shape if dim > 1]
            if axes:
                Hcand = max(axes)
                axis = list(s.shape).index(Hcand)
                moved = np.moveaxis(s, axis, 0)
                raw = moved.reshape(Hcand, -1).mean(axis=1).astype(np.float32)
            else:
                raw = np.array([float(np.mean(s))], dtype=np.float32)
        out = np.zeros((target_H,), dtype=np.float32)
        out[:min(target_H, len(raw))] = raw[:target_H]
        return out

    # ---------------- Prep ----------------
    ensure_dir(bin_dir)
    ensure_dir(hdr_dir)

    device = getattr(model, "device", next(model.parameters()).device)
    dtype = getattr(model, "dtype", next(model.parameters()).dtype)

    # ---------------- Collections ----------------
    # Aggregate QuantLinear stats across layers keyed by leaf name (e.g., q_proj)
    linear_scales_by_leaf: Dict[str, Dict[int, np.ndarray]] = {}
    linear_sums_by_leaf: Dict[str, Dict[int, np.ndarray]] = {}
    out_feats_by_leaf: Dict[str, int] = {}
    layers_seen_by_leaf: Dict[str, set] = {}

    # Fallback headers when a module has no layer index
    per_module_headers: List[Tuple[str, np.ndarray, np.ndarray]] = []

    # Attention per-head scales (Q/K/V/A)
    attn_scales = {"Q": {}, "K": {}, "V": {}, "A": {}}

    # RMSNorm weights matrix: rows = 2*decoder_layer_num + 1 (last row is model.norm)
    rms_total_rows = 2 * decoder_layer_num + 1
    RMS_weight_mat = np.zeros((rms_total_rows, hidden_dim), dtype=np.float32)
    rms_filled = np.zeros((rms_total_rows,), dtype=bool)

    # Map RMSNorm module names to fixed row indices in RMS_weight_mat
    def rms_row_index_from_name(name: str) -> Optional[int]:
        """
        Row mapping:
          - model.layers.<L>.input_layernorm          -> row = 2*L
          - model.layers.<L>.post_attention_layernorm -> row = 2*L + 1
          - model.norm                                -> row = 2*decoder_layer_num
        """
        m = re.search(r"\blayers\.(\d+)\.input_layernorm\b", name)
        if m:
            L = int(m.group(1))
            if 0 <= L < decoder_layer_num:
                return 2 * L
            return None
        m = re.search(r"\blayers\.(\d+)\.post_attention_layernorm\b", name)
        if m:
            L = int(m.group(1))
            if 0 <= L < decoder_layer_num:
                return 2 * L + 1
            return None
        if re.search(r"\bmodel\.norm\b", name) or name.endswith(".norm") or name == "norm":
            return 2 * decoder_layer_num
        return None

    # ---------------- Main pass ----------------
    for name, module in model.named_modules():

        # ===== QuantLinear =====
        if isinstance(module, qnn.QuantLinear):
            qw = module.quant_weight()               # QuantTensor
            w_val = qw.value                         # float [out_features, in_features]
            out_features, in_features = w_val.shape[-2], w_val.shape[-1]

            # Per-output-channel scale with shape [out_features]
            s_per_out = to_per_out_channel_scale(qw.scale, out_features).to(w_val.dtype).to(w_val.device)

            # Floating quantization before rounding/clamping; then store as int8
            wq_quant_float = w_val / s_per_out.view(-1, 1)
            wq_qint8 = torch.round(wq_quant_float).clamp_(-8, 7).to(torch.int8)  # range [-8..7]

            # 1) Write .bin (int8), row-major [out_features, in_features]
            layer_idx = get_layer_idx(name)
            leaf = get_leaf_name(name)
            bin_fname = f"{leaf}" + (f"_L{layer_idx:02d}" if layer_idx is not None else "")
            bin_path = os.path.join(bin_dir, f"{bin_fname}.bin")
            with open(bin_path, "wb") as f:
                f.write(wq_qint8.contiguous().view(-1).cpu().numpy().tobytes())

            # 2) Aggregate per-out scales and per-out sums of ORIGINAL float weights
            w_sum_fp32 = torch.sum(w_val, dim=1).to(torch.float32)

            s_np = s_per_out.detach().cpu().float().numpy()
            sum_np = w_sum_fp32.detach().cpu().float().numpy()

            if layer_idx is None:
                per_module_headers.append((name, s_np, sum_np))
            else:
                linear_scales_by_leaf.setdefault(leaf, {})
                linear_sums_by_leaf.setdefault(leaf, {})
                out_feats_by_leaf[leaf] = max(out_feats_by_leaf.get(leaf, 0), out_features)
                layers_seen_by_leaf.setdefault(leaf, set()).add(layer_idx)
                linear_scales_by_leaf[leaf][layer_idx] = s_np
                linear_sums_by_leaf[leaf][layer_idx] = sum_np

        # ===== QuantScaledDotProductAttention =====
        elif isinstance(module, qnn.QuantScaledDotProductAttention):
            # Derive head counts and per-head dims
            Hq = getattr(module, "num_heads", head_num)
            Hkv = getattr(module, "num_key_value_heads", kv_head_num)

            # If module exposes head_dim, use it for both; otherwise infer
            D_mod = getattr(module, "head_dim", None)
            if D_mod is not None:
                Dq = int(D_mod)
                Dkv = int(D_mod)
            else:
                Dq = max(1, hidden_dim // max(1, Hq))
                # If kv_hidden_dim not provided or inconsistent, fall back to Dq
                Dkv = max(1, kv_hidden_dim // max(1, Hkv)) if kv_hidden_dim else Dq

            S = 8  # small sequence length is sufficient

            # Probe tensors
            q = torch.randn(1, Hq, S, Dq, device=device, dtype=dtype)
            kT = torch.randn(1, Hkv, Dkv, S, device=device, dtype=dtype)
            scores = torch.randn(1, Hq, S, S, device=device, dtype=dtype)
            v = torch.randn(1, Hkv, S, Dkv, device=device, dtype=dtype)

            INTERESTED = {
                "attn-q_scaled_quant": ("Q", Hq),
                "attn-k_transposed_quant": ("K", Hkv),
                "attn-v_quant": ("V", Hkv),
                "attn-attn_output_weights_quant": ("A", Hq),
            }
            layer_idx = get_layer_idx(name)
            layer_bucket = layer_idx if layer_idx is not None else -1

            for ni, qi in module.named_modules():
                if isinstance(qi, qnn.QuantIdentity):
                    full_name = f"{name}-{ni}"
                    tag, target_H = None, None
                    for key, (short, targH) in INTERESTED.items():
                        if key in full_name:
                            tag, target_H = short, targH
                            break
                    if tag is None:
                        continue

                    rqt = qi.return_quant_tensor
                    qi.return_quant_tensor = True
                    if tag == "Q":
                        qa = qi(q)
                    elif tag == "K":
                        qa = qi(kT)
                    elif tag == "V":
                        qa = qi(v)
                    elif tag == "A":
                        qa = qi(scores)
                    qi.return_quant_tensor = rqt

                    s_np = qa.scale.detach().cpu().float().numpy()
                    per_head = per_head_from_scale_array(s_np, target_H)
                    attn_scales[tag].setdefault(layer_bucket, per_head)

        # ===== RMSNorm collection =====
        elif isinstance(module, nn.RMSNorm):
            row = rms_row_index_from_name(name)
            if row is not None and 0 <= row < 2 * decoder_layer_num + 1:
                if hasattr(module, "weight") and module.weight is not None:
                    w = module.weight.detach().cpu().float().numpy()  # [hidden_dim]
                    vec = np.zeros((hidden_dim,), dtype=np.float32)
                    vec[:min(hidden_dim, len(w))] = w[:hidden_dim]
                    RMS_weight_mat[row, :] = vec
                    rms_filled[row] = True
                # else: keep zeros (valid for no-affine RMSNorm)

        # ===== Embedding export =====
        if isinstance(module, nn.Embedding):
            W = module.weight.detach().cpu().float().contiguous().numpy()
            emb_bin_name = f"{sanitize_name(name)}_fp32.bin"
            emb_bin_path = os.path.join(bin_dir, emb_bin_name)
            with open(emb_bin_path, "wb") as f:
                f.write(W.tobytes())

    # ---------------- Write QuantLinear headers ----------------
    def write_linear_headers():
        for leaf, by_layer in linear_scales_by_leaf.items():
            outF = out_feats_by_leaf[leaf]
            S_mat = np.zeros((decoder_layer_num, outF), dtype=np.float32)
            SUM_mat = np.zeros((decoder_layer_num, outF), dtype=np.float32)

            for L, s in by_layer.items():
                if 0 <= L < decoder_layer_num:
                    sm = linear_sums_by_leaf[leaf][L]
                    S_mat[L, :min(outF, len(s))] = s[:outF]
                    SUM_mat[L, :min(outF, len(sm))] = sm[:outF]

            base = f"w_{leaf}"  # e.g., w_q_proj
            guard = header_guard(f"{base}_s_sum")
            hdr_path = os.path.join(hdr_dir, f"{base}_s_sum.h")
            with open(hdr_path, "w") as f:
                f.write(f"#ifndef {guard}\n#define {guard}\n\n")
                f.write(c_array_2d_float(f"{base}_s", S_mat))
                f.write("\n\n")
                f.write(c_array_2d_float(f"{base}_sum", SUM_mat))
                f.write("\n\n#endif // " + guard + "\n")

        # Per-module fallback headers (no layer index)
        for mod_full, s_np, sum_np in per_module_headers:
            base = f"w_{get_leaf_name(mod_full)}_{sanitize_name(mod_full)}"
            guard = header_guard(base)
            hdr_path = os.path.join(hdr_dir, f"{base}.h")
            with open(hdr_path, "w") as f:
                f.write(f"#ifndef {guard}\n#define {guard}\n\n")
                f.write("static const int NUM_CHANNELS = " + str(len(s_np)) + ";\n\n")
                f.write(
                    "static const float "
                    + base
                    + "_s[NUM_CHANNELS] = { "
                    + ", ".join(f"{float(v):.9g}" for v in s_np)
                    + " };\n\n"
                )
                f.write(
                    "static const float "
                    + base
                    + "_sum[NUM_CHANNELS] = { "
                    + ", ".join(f"{float(v):.9g}" for v in sum_np)
                    + " };\n\n"
                )
                f.write("#endif // " + guard + "\n")

    # ---------------- Write attention headers ----------------
    def write_attn_header(tag: str, data: Dict[int, np.ndarray], fname: str, target_heads: int):
        # Build fixed [decoder_layer_num, target_heads] and ignore layer_bucket == -1
        M = np.zeros((decoder_layer_num, target_heads), dtype=np.float32)
        for L, per_head in data.items():
            if 0 <= L < decoder_layer_num:
                ph = np.asarray(per_head, dtype=np.float32)
                M[L, :min(target_heads, len(ph))] = ph[:target_heads]

        macro_guard = f"{fname.upper()}_H_"
        path = os.path.join(hdr_dir, f"{fname}.h")
        with open(path, "w") as f:
            f.write(f"#ifndef {macro_guard}\n#define {macro_guard}\n\n")
            f.write(c_array_2d_float(f"{tag}_s", M))
            f.write("\n\n#endif // " + macro_guard + "\n")

    # ---------------- Write RMSNorm header ----------------
    def write_rmsnorm_header():
        base = "w_rmsnorm"
        guard = header_guard(base)
        hdr_path = os.path.join(hdr_dir, f"{base}.h")
        with open(hdr_path, "w") as f:
            f.write(f"#ifndef {guard}\n#define {guard}\n\n")
            f.write(c_array_2d_float("RMSNorm_weight", RMS_weight_mat))
            f.write("\n\n#endif // " + guard + "\n")

    # ---------------- Execute writes ----------------
    write_linear_headers()
    write_attn_header("Q", attn_scales["Q"], "Q_s", head_num)
    write_attn_header("K", attn_scales["K"], "K_s", kv_head_num)
    write_attn_header("V", attn_scales["V"], "V_s", kv_head_num)
    write_attn_header("A", attn_scales["A"], "A_s", head_num)
    write_rmsnorm_header()

    # Optional diagnostics for missing RMSNorm rows
    if not rms_filled.all():
        missing_idx = np.nonzero(~rms_filled)[0].tolist()
        print(f"[WARN] Some RMSNorm rows were not filled (left as zeros): {missing_idx}")

    print(f"[OK] Saved QuantLinear int8 bins to: {bin_dir}")
    print(f"[OK] Saved Embedding fp32 bins to: {bin_dir}")
    print(f"[OK] Saved headers to: {hdr_dir}")


# Example usage (after the model is fully loaded and on the desired device/dtype):
# export_parameters(
#     model,
#     decoder_layer_num=16,
#     head_num=16,          # Q/A
#     kv_head_num=8,        # K/V
#     hidden_dim=2048,
#     kv_hidden_dim=512,
# )
