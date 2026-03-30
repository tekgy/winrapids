"""
Prototype: Tiled .tb file format with streaming and predicate pushdown.
Tests whether tile-based architecture gives us "something for nothing."
"""
import cupy as cp
import numpy as np
import time
import struct
import os
import cupyx

TILE_ROWS = 65536
HEADER_SIZE = 4096
TILE_HEADER_SIZE = 64

def write_tb_file(path, columns, tile_rows=TILE_ROWS):
    n_rows = len(columns[0])
    n_cols = len(columns)
    n_tiles = (n_rows + tile_rows - 1) // tile_rows

    with open(path, 'wb') as f:
        header = bytearray(HEADER_SIZE)
        struct.pack_into('4s', header, 0, b'TBFM')
        struct.pack_into('I', header, 4, 1)
        struct.pack_into('Q', header, 8, n_rows)
        struct.pack_into('I', header, 16, n_cols)
        struct.pack_into('I', header, 20, tile_rows)
        struct.pack_into('I', header, 24, n_tiles)
        f.write(header)

        for tile_idx in range(n_tiles):
            start = tile_idx * tile_rows
            end = min(start + tile_rows, n_rows)
            actual_rows = end - start

            tile_header = bytearray(TILE_HEADER_SIZE)
            struct.pack_into('I', tile_header, 0, actual_rows)
            col0_slice = columns[0][start:end]
            struct.pack_into('d', tile_header, 8, float(np.min(col0_slice)))
            struct.pack_into('d', tile_header, 16, float(np.max(col0_slice)))
            struct.pack_into('d', tile_header, 24, float(np.sum(col0_slice)))
            struct.pack_into('I', tile_header, 32, actual_rows)
            f.write(tile_header)

            for col in columns:
                chunk = col[start:end].astype(np.float64)
                raw = chunk.tobytes()
                padding = (64 - len(raw) % 64) % 64
                f.write(raw)
                if padding:
                    f.write(b'\x00' * padding)

    return os.path.getsize(path)

def read_tile_headers(path):
    headers = []
    with open(path, 'rb') as f:
        gh = f.read(HEADER_SIZE)
        n_rows = struct.unpack_from('Q', gh, 8)[0]
        n_cols = struct.unpack_from('I', gh, 16)[0]
        tile_rows = struct.unpack_from('I', gh, 20)[0]
        n_tiles = struct.unpack_from('I', gh, 24)[0]

        for _ in range(n_tiles):
            th = f.read(TILE_HEADER_SIZE)
            rows = struct.unpack_from('I', th, 0)[0]
            headers.append({
                'rows': rows,
                'min': struct.unpack_from('d', th, 8)[0],
                'max': struct.unpack_from('d', th, 16)[0],
                'sum': struct.unpack_from('d', th, 24)[0],
                'count': struct.unpack_from('I', th, 32)[0],
            })
            col_bytes = rows * 8
            aligned = col_bytes + (64 - col_bytes % 64) % 64
            f.seek(aligned * n_cols, 1)

    return headers, n_cols

def main():
    np.random.seed(42)
    n_rows = 1_000_000
    n_cols = 5
    n_groups = 4600

    print(f"Generating {n_rows:,} rows x {n_cols} columns...")
    prices = 100 + np.cumsum(np.random.randn(n_rows) * 0.1).astype(np.float64)
    volumes = np.random.exponential(1000, n_rows).astype(np.float64)
    keys = np.random.randint(0, n_groups, n_rows).astype(np.float64)
    col3 = np.random.randn(n_rows).astype(np.float64)
    col4 = np.random.randn(n_rows).astype(np.float64)
    columns = [prices, volumes, keys, col3, col4]

    tb_path = "R:/winrapids/experiments/test_tiled.tb"
    t0 = time.perf_counter()
    file_size = write_tb_file(tb_path, columns)
    t_write = (time.perf_counter() - t0) * 1000
    n_tiles = (n_rows + TILE_ROWS - 1) // TILE_ROWS
    print(f"Wrote {file_size/1e6:.1f} MB in {t_write:.1f} ms ({n_tiles} tiles)")
    print()

    # === Test 1: Header-only aggregation ===
    print("=== Test 1: Global stats from headers only (ZERO data read) ===")
    t0 = time.perf_counter()
    headers, nc = read_tile_headers(tb_path)
    global_sum = sum(h['sum'] for h in headers)
    global_count = sum(h['count'] for h in headers)
    global_mean = global_sum / global_count
    global_min = min(h['min'] for h in headers)
    global_max = max(h['max'] for h in headers)
    t_header = (time.perf_counter() - t0) * 1000

    bytes_read = HEADER_SIZE + n_tiles * TILE_HEADER_SIZE
    data_bytes = n_rows * n_cols * 8

    print(f"  Time: {t_header:.3f} ms")
    print(f"  Bytes read: {bytes_read:,} ({bytes_read/1024:.1f} KB)")
    print(f"  Data bytes: {data_bytes:,} ({data_bytes/1e6:.1f} MB)")
    print(f"  Data SKIPPED: {(1 - bytes_read/data_bytes)*100:.2f}%")
    print(f"  Sum err: {abs(global_sum - np.sum(prices)) / abs(np.sum(prices)):.2e}")
    print(f"  Min/Max correct: {global_min == np.min(prices) and global_max == np.max(prices)}")
    print()

    # === Test 2: Predicate pushdown ===
    print("=== Test 2: Predicate pushdown (price > 110) ===")
    threshold = 110.0
    tiles_needed = [i for i, h in enumerate(headers) if h['max'] > threshold]
    tiles_skipped = n_tiles - len(tiles_needed)
    print(f"  Tiles needed: {len(tiles_needed)}/{n_tiles} ({len(tiles_needed)/n_tiles*100:.0f}%)")
    print(f"  Tiles SKIPPED: {tiles_skipped} ({tiles_skipped/n_tiles*100:.0f}%)")
    print(f"  Data skipped: {tiles_skipped * TILE_ROWS * n_cols * 8 / 1e6:.1f} MB")
    print()

    # === Test 3: Tiled streaming GroupBy ===
    print("=== Test 3: Tiled streaming GroupBy ===")
    group_sums = cp.zeros(n_groups, dtype=cp.float64)

    t0 = time.perf_counter()
    with open(tb_path, 'rb') as f:
        f.read(HEADER_SIZE)
        for tile_idx in range(n_tiles):
            f.read(TILE_HEADER_SIZE)
            actual_rows = min(TILE_ROWS, n_rows - tile_idx * TILE_ROWS)
            col_bytes = actual_rows * 8
            aligned = col_bytes + (64 - col_bytes % 64) % 64

            price_data = np.frombuffer(f.read(col_bytes), dtype=np.float64)
            if aligned > col_bytes:
                f.seek(aligned - col_bytes, 1)
            f.seek(aligned, 1)  # skip volume
            ticker_data = np.frombuffer(f.read(col_bytes), dtype=np.float64)
            if aligned > col_bytes:
                f.seek(aligned - col_bytes, 1)
            for _ in range(n_cols - 3):
                f.seek(aligned, 1)

            price_gpu = cp.asarray(price_data)
            ticker_gpu = cp.asarray(ticker_data).astype(cp.int64)
            cupyx.scatter_add(group_sums, ticker_gpu, price_gpu)

    cp.cuda.Device(0).synchronize()
    t_tiled = (time.perf_counter() - t0) * 1000

    # Full load comparison
    t0 = time.perf_counter()
    all_prices = cp.asarray(prices)
    all_tickers = cp.asarray(keys.astype(np.int64))
    group_sums_full = cp.zeros(n_groups, dtype=cp.float64)
    cupyx.scatter_add(group_sums_full, all_tickers, all_prices)
    cp.cuda.Device(0).synchronize()
    t_full = (time.perf_counter() - t0) * 1000

    max_err = float(cp.max(cp.abs(group_sums - group_sums_full)))

    print(f"  Tiled streaming: {t_tiled:.1f} ms")
    print(f"  Full load:       {t_full:.1f} ms")
    print(f"  Correctness:     max_err = {max_err:.2e}")
    print(f"  Tile memory:     {TILE_ROWS * 8 * 2 / 1e6:.1f} MB")
    print(f"  Full memory:     {n_rows * 8 * 2 / 1e6:.1f} MB")
    print(f"  Memory savings:  {(1 - TILE_ROWS * 2 / (n_rows * 2)) * 100:.0f}%")
    print()

    # === Test 4: Pushdown + streaming GroupBy ===
    print("=== Test 4: Pushdown + streaming GroupBy ===")
    group_sums_pushed = cp.zeros(n_groups, dtype=cp.float64)
    tiles_read = 0

    t0 = time.perf_counter()
    with open(tb_path, 'rb') as f:
        f.read(HEADER_SIZE)
        for tile_idx in range(n_tiles):
            th = f.read(TILE_HEADER_SIZE)
            actual_rows = min(TILE_ROWS, n_rows - tile_idx * TILE_ROWS)
            col0_max = struct.unpack_from('d', th, 16)[0]
            col_bytes = actual_rows * 8
            aligned = col_bytes + (64 - col_bytes % 64) % 64

            if col0_max <= threshold:
                f.seek(aligned * n_cols, 1)
                continue

            tiles_read += 1
            price_data = np.frombuffer(f.read(col_bytes), dtype=np.float64)
            if aligned > col_bytes:
                f.seek(aligned - col_bytes, 1)
            f.seek(aligned, 1)
            ticker_data = np.frombuffer(f.read(col_bytes), dtype=np.float64)
            if aligned > col_bytes:
                f.seek(aligned - col_bytes, 1)
            for _ in range(n_cols - 3):
                f.seek(aligned, 1)

            price_gpu = cp.asarray(price_data)
            ticker_gpu = cp.asarray(ticker_data).astype(cp.int64)
            mask = (price_gpu > threshold).astype(cp.float64)
            cupyx.scatter_add(group_sums_pushed, ticker_gpu, price_gpu * mask)

    cp.cuda.Device(0).synchronize()
    t_pushed = (time.perf_counter() - t0) * 1000

    print(f"  With pushdown:    {t_pushed:.1f} ms")
    print(f"  Without pushdown: {t_tiled:.1f} ms")
    print(f"  Speedup:          {t_tiled/max(t_pushed, 0.01):.1f}x")
    print(f"  Tiles read:       {tiles_read}/{n_tiles}")
    print()

    os.remove(tb_path)

    print("=== SUMMARY ===")
    print(f"  Header-only agg: {t_header:.3f} ms (read {bytes_read/1024:.1f} KB of {data_bytes/1e6:.1f} MB)")
    print(f"  Tiled streaming: {t_tiled:.1f} ms (constant memory)")
    print(f"  With pushdown:   {t_pushed:.1f} ms (skip non-matching tiles)")
    print(f"  Full load:       {t_full:.1f} ms (needs all data in GPU)")
    print()
    print("  Tam doesn't load the file. Tam streams the tiles he needs.")

if __name__ == "__main__":
    main()
