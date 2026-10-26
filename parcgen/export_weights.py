"""Pull PARC's motion generator weights out of its PyTorch checkpoint into a plain .npz.

    python3 parcgen/export_weights.py model_40000.ckpt parcgen/parc_mdm.npz [--ema]

Run with PyTorch installed. The checkpoint is loaded with weights_only=True, which only rebuilds
tensors and plain containers, so it cannot run code; nothing else in it is used. Writes float32
arrays (the denoiser's weights and the feature mean/std) and the config values the generator needs.
"""
import argparse
import json

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('ckpt')
ap.add_argument('out')
ap.add_argument('--ema', action='store_true', help='the EMA weights instead of the trained ones (PARC loads the trained ones)')
a = ap.parse_args()
ck = torch.load(a.ckpt, weights_only=True, map_location='cpu')
sd = ck['ema_model' if a.ema else 'model']
arrays = {k: v.numpy().astype(np.float32) for k, v in sd.items()}
arrays['features_mean'] = ck['extra']['_mdm_features_mean'].numpy()
arrays['features_std'] = ck['extra']['_mdm_features_std'].numpy()
keep = ['diffusion_timesteps', 'd_model', 'num_heads', 'd_hid', 'num_layers', 'num_prev_states', 'seq_len',
        'sequence_fps', 'heightmap', 'target_dir_len_eps', 'target_dir_heading_eps', 'relative_z_style', 'features']
cfg = {k: ck['cfg'][k] for k in keep if k in ck['cfg']}
cfg['seq_len'] = ck['extra']['_seq_len']
arrays['config_json'] = np.frombuffer(json.dumps(cfg).encode(), np.uint8)
np.savez(a.out, **arrays)

# The same weights as one float32 blob in the order parcgen.c binds them, for the C runtime.
ORDER = [f'_obs_conv_net._net.{i}.{p}' for i in (0, 2, 4, 6) for p in ('weight', 'bias')] + [
    '_obs_embed.weight', '_obs_embed.bias', '_in_mlp_target.model.0.weight', '_in_mlp_target.model.0.bias',
    '_in_mlp_target.model.3.weight', '_in_mlp_target.model.3.bias', '_embed_prev_state_noise_indicator.weight', '_in_pos_encoding.pe'] + [
    f'_transformer_encoder.layers.{l}.{p}' for l in range(4) for p in (
        'self_attn.in_proj_weight', 'self_attn.in_proj_bias', 'self_attn.out_proj.weight', 'self_attn.out_proj.bias',
        'linear1.weight', 'linear1.bias', 'linear2.weight', 'linear2.bias', 'norm1.weight', 'norm1.bias', 'norm2.weight', 'norm2.bias')] + [
    '_out_mlp.model.0.weight', '_out_mlp.model.0.bias', '_out_mlp.model.3.weight', '_out_mlp.model.3.bias',
    '_embed_t.sequence_pos_encoder.pe', '_embed_t.time_embed.0.weight', '_embed_t.time_embed.0.bias',
    '_embed_t.time_embed.2.weight', '_embed_t.time_embed.2.bias', '_in_mlp_gen_seq.model.0.weight', '_in_mlp_gen_seq.model.0.bias',
    '_in_mlp_gen_seq.model.3.weight', '_in_mlp_gen_seq.model.3.bias', 'features_mean', 'features_std']
assert set(ORDER) == set(arrays) - {'config_json'}, set(arrays) ^ set(ORDER)
blob = np.concatenate([arrays[k].ravel() for k in ORDER]).astype('<f4')
blob.tofile(a.out.replace('.npz', '.bin'))
print(f"{a.out.replace('.npz', '.bin')}: {blob.size} floats")
print(f'{a.out}: {sum(v.size for k, v in arrays.items() if k != "config_json")} floats', cfg)
