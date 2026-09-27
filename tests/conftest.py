import os
import sys
from pathlib import Path

os.environ.setdefault('XP_RUNTIME', 'CPU')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import struct
import numpy as np
import pytest


@pytest.fixture
def checkpoint_factory(tmp_path):
    from backend import AMP_TYPE
    from checkpoint import model_config, tensor_specs

    def create(dtype='BF16', alias_delta=0):
        directory = tmp_path / (dtype + str(alias_delta))
        directory.mkdir()
        config = dict(model_type='gemma4_unified_text', vocab_size=16,
                      hidden_size=8, intermediate_size=16, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=4,
                      global_head_dim=8, max_position_embeddings=12, sliding_window=4)
        (directory / 'config.json').write_text(json.dumps(config))
        specs = tensor_specs(model_config(config))
        values = {}
        rng = np.random.default_rng(9)
        storage_dtype = {'BF16': AMP_TYPE, 'F16': np.float16, 'F32': np.float32}[dtype]
        for name,spec in specs.items():
            values[name] = (np.ones(spec.shape) if spec.target[-1] in ('gamma','layer_scalar')
                            else rng.normal(size=spec.shape)*.2).astype(storage_dtype)
        values['lm_head.weight'] = values['embed_tokens.weight'].copy()
        values['lm_head.weight'][0,0] += alias_delta
        header, buffers, offset = {}, [], 0
        for name,array in values.items():
            raw = array.tobytes()
            header[name] = dict(dtype=dtype,shape=list(array.shape),data_offsets=[offset,offset+len(raw)])
            buffers.append(raw); offset += len(raw)
        encoded = json.dumps(header).encode()
        encoded += b' ' * (-len(encoded) % 8)
        (directory / 'model.safetensors').write_bytes(struct.pack('<Q',len(encoded))+encoded+b''.join(buffers))
        return directory, values
    return create
