"""
Checksum every weight tensor in a DCP shard, mapped to its full tensor key name.

Usage:
    python read_dcp_metadata.py  <.metadata>  <shard_file>
    python read_dcp_metadata.py  <.metadata>  <shard_file>  --debug   # dump raw metadata structure
"""

import sys
import os
import pickle
import hashlib
import torch


def tensor_sha256(t):
    raw = t.contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()[:16]


def load_meta(meta_path):
    with open(meta_path, "rb") as f:
        return pickle.load(f)


def debug_meta(meta):
    """Dump the raw metadata structure so we can see what attributes exist."""
    print("=== Metadata top-level attrs ===")
    print([a for a in dir(meta) if not a.startswith("__")])

    sd = meta.state_dict_metadata
    first_key = next(iter(sd))
    val = sd[first_key]
    print(f"\n=== state_dict_metadata['{first_key}'] ===")
    print("type:", type(val))
    print("attrs:", [a for a in dir(val) if not a.startswith("__")])
    for a in dir(val):
        if not a.startswith("__"):
            try:
                print(f"  .{a} =", getattr(val, a))
            except Exception:
                pass

    if hasattr(val, "chunks") and val.chunks:
        chunk = val.chunks[0]
        print(f"\n=== chunks[0] ===")
        print("type:", type(chunk))
        print("attrs:", [a for a in dir(chunk) if not a.startswith("__")])
        for a in dir(chunk):
            if not a.startswith("__"):
                try:
                    print(f"  .{a} =", getattr(chunk, a))
                except Exception:
                    pass

    if hasattr(meta, "storage_data") and meta.storage_data:
        idx = next(iter(meta.storage_data))
        loc = meta.storage_data[idx]
        print(f"\n=== storage_data (first entry) ===")
        print("key type:", type(idx), "| val type:", type(loc))
        print("key attrs:", [a for a in dir(idx) if not a.startswith("__")])
        print("val attrs:", [a for a in dir(loc) if not a.startswith("__")])
        for a in dir(loc):
            if not a.startswith("__"):
                try:
                    print(f"  loc.{a} =", getattr(loc, a))
                except Exception:
                    pass


def build_chunk_index(meta, shard_fname):
    """
    Returns ordered list of (tensor_key, dtype, chunk) for chunks in shard_fname.
    Tries three strategies across different PyTorch DCP versions.
    """

    # --- Strategy 1: meta.storage_data (PyTorch >= 2.1 standard) ---
    # storage_data: Dict[MetadataIndex, StorageInfo]
    # MetadataIndex has .fqn (tensor key) and .index (chunk position in that tensor)
    if hasattr(meta, "storage_data") and meta.storage_data:
        index = {}
        for meta_idx, storage_loc in meta.storage_data.items():
            fname = (
                getattr(storage_loc, "relative_path", None)
                or getattr(storage_loc, "filename", None)
                or getattr(storage_loc, "path", None)
                or str(storage_loc)
            )
            fname = os.path.basename(str(fname))
            if fname != shard_fname:
                continue
            key = getattr(meta_idx, "fqn", str(meta_idx))
            chunk_pos = getattr(meta_idx, "index", None)

            val = meta.state_dict_metadata.get(key)
            if val is None or not hasattr(val, "chunks"):
                continue
            if chunk_pos is not None and chunk_pos < len(val.chunks):
                chunk = val.chunks[chunk_pos]
            else:
                chunk = val.chunks[0]
            dtype = val.properties.dtype
            byte_offset = getattr(storage_loc, "offset", 0)
            index[byte_offset] = (key, dtype, chunk)

        if index:
            return [v for _, v in sorted(index.items())]

    # --- Strategy 2: chunk.storage_index.filename (PyTorch 2.2+) ---
    result = []
    for key, val in meta.state_dict_metadata.items():
        if not hasattr(val, "chunks"):
            continue
        dtype = val.properties.dtype
        for chunk in val.chunks:
            si = getattr(chunk, "storage_index", None)
            if si is None:
                continue
            fname = (
                getattr(si, "filename", None)
                or getattr(si, "relative_path", None)
                or str(si)
            )
            if os.path.basename(str(fname)) == shard_fname:
                byte_offset = getattr(si, "offset", 0)
                result.append((byte_offset, key, dtype, chunk))

    if result:
        return [(k, d, c) for _, k, d, c in sorted(result)]

    # --- Strategy 3: no file mapping — just return ALL chunks in metadata order ---
    # (works when there's only one shard or user passes the right shard)
    print(f"[warn] Could not match chunks to '{shard_fname}' via metadata.")
    print(f"[warn] Falling back: listing all chunks in metadata order.\n")
    result = []
    for key, val in meta.state_dict_metadata.items():
        if not hasattr(val, "chunks"):
            continue
        dtype = val.properties.dtype
        for chunk in val.chunks:
            result.append((key, dtype, chunk))
    return result


def load_shard_tensors(shard_path):
    """Load a DCP shard; return list of tensors in chunk order."""
    raw = torch.load(shard_path, map_location="cpu", weights_only=False)

    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, torch.Tensor)]
    elif isinstance(raw, dict):
        return [v for v in raw.values() if isinstance(v, torch.Tensor)]
    elif isinstance(raw, torch.Tensor):
        return [raw]
    else:
        print(f"[warn] Unknown shard type: {type(raw)}")
        return []


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    debug = "--debug" in sys.argv

    if len(args) < 2:
        print(__doc__)
        sys.exit(1)

    meta_path, shard_path = args[0], args[1]
    shard_fname = os.path.basename(shard_path)

    meta = load_meta(meta_path)

    if debug:
        debug_meta(meta)
        print("\n" + "="*70 + "\n")

    chunks = build_chunk_index(meta, shard_fname)
    tensors = load_shard_tensors(shard_path)

    if not tensors:
        print("No tensors loaded from shard. Try --debug to inspect structure.")
        sys.exit(1)

    print(f"{'Tensor key':<80}  {'shape':<22}  {'dtype':<18}  sha256[:16]")
    print(f"{'-'*80}  {'-'*22}  {'-'*18}  {'-'*16}")

    for i, (key, dtype, chunk) in enumerate(chunks):
        if i >= len(tensors):
            print(f"{key:<80}  (chunk {i} not in shard)")
            continue

        t = tensors[i]

        # reinterpret raw bytes into the correct dtype + shape
        target_shape = tuple(chunk.sizes)
        if t.dtype == torch.uint8 and dtype != torch.uint8:
            try:
                t = t.view(dtype).reshape(target_shape)
            except Exception:
                pass
        else:
            try:
                t = t.reshape(target_shape)
            except Exception:
                pass

        chk = tensor_sha256(t)
        print(f"{key:<80}  {str(tuple(t.shape)):<22}  {str(t.dtype):<18}  {chk}")


if __name__ == "__main__":
    main()
