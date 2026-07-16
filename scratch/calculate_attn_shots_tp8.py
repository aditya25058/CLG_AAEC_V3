import sys

_ATTN_CHUNK_START = 16
_ATTN_N_DECODE_START = 1
_ATTN_KV_START = 16
_BLOCK_SIZE = 16

def _geometric_grid(max_value: int, start: int, factor: float = 2.0) -> list[int]:
    if factor <= 1.0:
        raise ValueError(f"factor must be > 1.0; got {factor}")
    if max_value < start:
        return [0]
    values: list[int] = [0, start]
    v: float = float(start)
    while True:
        v *= factor
        iv = int(round(v))
        if iv > max_value:
            break
        if iv != values[-1]:
            values.append(iv)
    if values[-1] != max_value:
        values.append(max_value)
    return values

class Limits:
    max_num_batched_tokens = 2048
    max_num_seqs = 256
    max_model_len = 163840
    num_cache_tokens = 1141840

limits = Limits()
attention_max_kv = 16384
attention_chunk_factor = 2.0
attention_kv_factor = 2.0

chunk_vals = _geometric_grid(
    limits.max_num_batched_tokens, _ATTN_CHUNK_START,
    factor=attention_chunk_factor,
)
n_dec_vals = _geometric_grid(
    limits.max_num_seqs, _ATTN_N_DECODE_START,
)
kv_cap = min(attention_max_kv, limits.max_model_len)
kv_vals = _geometric_grid(
    kv_cap, _ATTN_KV_START, factor=attention_kv_factor,
)

print(f"chunk_vals: {chunk_vals}")
print(f"n_dec_vals: {n_dec_vals}")
print(f"kv_vals: {kv_vals}")

shots = []
for chunk in chunk_vals:
    for kv_p in kv_vals:
        if chunk == 0 and kv_p != 0:
            continue
        for n_dec in n_dec_vals:
            for kv_d in kv_vals:
                if n_dec == 0 and kv_d != 0:
                    continue
                if n_dec > 0 and kv_d == 0:
                    continue
                if chunk == 0 and n_dec == 0:
                    continue
                
                if chunk + n_dec > (
                    limits.max_num_batched_tokens + limits.max_num_seqs
                ):
                    continue
                n_reqs = (1 if chunk > 0 else 0) + n_dec
                if n_reqs > limits.max_num_seqs:
                    continue
                if chunk > 0 and chunk + kv_p + 1 > limits.max_model_len:
                    continue
                if n_dec > 0 and 1 + kv_d + 1 > limits.max_model_len:
                    continue
                def _aligned(total_len: int) -> int:
                    return ((total_len + _BLOCK_SIZE - 1) // _BLOCK_SIZE) * _BLOCK_SIZE
                prefill_block_toks = (
                    _aligned(chunk + kv_p) if chunk > 0 else 0
                )
                decode_block_toks = (
                    n_dec * _aligned(1 + kv_d) if n_dec > 0 else 0
                )
                if (prefill_block_toks + decode_block_toks > limits.num_cache_tokens):
                    continue
                shots.append((chunk, kv_p, n_dec, kv_d))

print(f"Total shots: {len(shots)}")
for i, shot in enumerate(shots[:100]):
    print(f"Shot {i+1}: {shot}")
