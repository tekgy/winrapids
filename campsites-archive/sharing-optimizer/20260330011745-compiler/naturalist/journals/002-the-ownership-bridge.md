# The Ownership Bridge

*Naturalist journal — 2026-03-30*

---

## The problem

The store tracks GPU buffers as raw `u64` device pointers (`BufferPtr`). The scan engine needs cudarc's `CudaSlice<f64>` to launch kernels. These are different ownership models:
- `BufferPtr`: a number. No ownership semantics. No lifetime. A phone number.
- `CudaSlice<f64>`: an RAII wrapper. Owns GPU memory. Freed on drop.

The compiler's execution plan routes `BufferPtr` values between nodes. When a MISS node dispatches a scan kernel, it needs to:
1. Pass the input `BufferPtr` to the scan engine (without taking ownership — the store or a previous node owns it)
2. Receive an output buffer (taking ownership — someone must keep it alive for downstream nodes)
3. Extract a `BufferPtr` from the output (for the store to track)

Three crossings of the ownership boundary. Each is a potential use-after-free or double-free.

## The solution

`CudaKernelDispatcher` (`cuda_dispatch.rs`) solves all three:

**Input crossing** — `ManuallyDrop`:
```rust
let input_dev = ManuallyDrop::new(
    self.stream.upgrade_device_ptr::<f64>(input_ptr, n)
);
```
Wraps the raw `u64` in a `CudaSlice` for the kernel launch, then `ManuallyDrop` prevents the `CudaSlice` destructor from freeing the caller's buffer. The input survives the scan untouched. Clean.

**Output crossing** — `ScanDeviceOutput`:
```rust
pub struct ScanDeviceOutput {
    primary: CudaSlice<f64>,    // owned output buffer
    primary_len: usize,
    secondary: Option<CudaSlice<f64>>,
    stream: Arc<CudaStream>,
}
```
The scan engine allocates output buffers and returns them in a struct that OWNS them. The `CudaSlice` destructor will free the GPU memory when `ScanDeviceOutput` is dropped.

**Lifetime crossing** — `Vec<ScanDeviceOutput>`:
```rust
pub struct CudaKernelDispatcher {
    engine: ScanEngine,
    scan_outputs: Vec<ScanDeviceOutput>,  // kept alive
}
```
The dispatcher accumulates all output buffers in a Vec. They live as long as the dispatcher lives. Downstream nodes read the raw `u64` pointers (via `primary_device_ptr()`) knowing the buffer is alive because the dispatcher hasn't been dropped yet.

When the dispatcher drops (pipeline execution ends), all GPU buffers are freed in one sweep. The store's `BufferPtr` entries become dangling — but the store doesn't own the memory, so it's the caller's responsibility to ensure consistency. In practice, the dispatcher lifetime encompasses the entire pipeline execution.

## What this confirms

**Observation #8 was right**: "the store is a phonebook — it maps identities to addresses. It doesn't own the buildings at those addresses." The `CudaKernelDispatcher` owns the buildings. The store tracks the addresses. The `BufferPtr` is the business card that connects them.

**The three-layer ownership model**:
- **Store** (`GpuStore`): metadata owner. Tracks provenance → pointer mapping. No GPU memory ownership.
- **Dispatcher** (`CudaKernelDispatcher`): GPU memory owner. Allocates, launches, keeps buffers alive.
- **Execution plan** (`execute()`): the orchestrator. Reads from store, dispatches to dispatcher, registers results back to store.

This separation is clean. The store doesn't need to know about cudarc. The dispatcher doesn't need to know about provenance. The execution plan connects them via `BufferPtr` — the thin bridge type.

## The ManuallyDrop pattern

`ManuallyDrop` is doing something subtle: it converts a non-owning `u64` into an owning `CudaSlice<f64>`, then suppresses the ownership. The intermediate `CudaSlice` is a lie — it claims to own memory it doesn't. `ManuallyDrop` is the escape hatch that makes the lie safe.

This pattern appears in FFI boundaries: wrap a foreign pointer in a Rust type for API compatibility, then `ManuallyDrop` to prevent Rust from cleaning up what it doesn't own. The Rust/CUDA boundary is an FFI boundary, even though both sides are "our" code.

The alternative — making `scan_device_ptr` accept raw `u64` directly — would require the scan engine to do unsafe pointer arithmetic internally. `ManuallyDrop` pushes the unsafety to one well-documented site.

## The sizeof validation is now exercised

My sizeof validation in `ensure_module()` runs automatically on this path. When `CudaKernelDispatcher` calls `self.engine.scan_device_ptr(op, input_ptr, n)`, the engine calls `ensure_module(op)`, which runs `query_sizeof` on first use. The pathmaker's Test 8 (real GPU scan with AddOp) would have triggered the sizeof check. It passed — confirming `sizeof(double) == 8` at runtime on the actual Blackwell GPU.

When WelfordOp or EWMOp first runs through the real GPU path, their sizeof checks will fire automatically.

---

*The store tracks identity. The dispatcher tracks ownership. The execution plan connects them. Three concerns, three types, one bridge (`BufferPtr`). The ManuallyDrop pattern is the ownership adapter — it lets Rust's type system see raw GPU pointers without claiming them. The Halide separation (algorithm/schedule) extends to ownership: the algorithm (store) says what. The schedule (dispatcher) says where and when. The execution plan binds them.*
