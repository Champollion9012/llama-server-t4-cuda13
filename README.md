# llama-server-t4-cuda13

Self-contained `llama-server` builds for running GGUF models on a **Google Colab Tesla T4**.

## Why this exists

Colab's runtime moved to **CUDA 13**. Prebuilt `llama-server` binaries compiled against CUDA 12
die before they even start:

```
llama-server: error while loading shared libraries: libcudart.so.12:
              cannot open shared object file: No such file or directory
```

These builds sidestep the problem entirely by statically linking the CUDA runtime, so the binary
depends on **no CUDA shared library at all**. It keeps working whatever CUDA version the host
happens to ship.

## What's in a release

A single stripped `llama-server` executable, ~601 MB, gzipped to ~412 MB.

| Property | Value |
|---|---|
| Dynamic dependencies | `libcuda.so.1` (driver), `libm.so.6`, `libc.so.6`, loader — **that's all** |
| CUDA runtime | statically linked (`cudart_static`, `cublas_static`, `cublasLt_static`) |
| C++ runtime | statically linked (`-static-libstdc++ -static-libgcc`) |
| glibc requirement | ≥ 2.34 |
| GPU architecture | `sm_75` real SASS — **Turing / T4 only** |
| CPU baseline | x86-64 with SSE4.2 / AVX / AVX2 / FMA / F16C, no AVX-512 |

## Compatibility warning

`CMAKE_CUDA_ARCHITECTURES=75-real` emits SASS only, with no PTX fallback. These binaries run on
**compute capability 7.5 only**. Colab occasionally allocates an L4 (8.9) or A100 (8.0) instead of
a T4; on those, this binary fails with
`no kernel image is available for execution on the device`.

Check before use:

```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader   # must print 7.5
```

Anything else — fall back to an official
[ggml-org/llama.cpp release](https://github.com/ggml-org/llama.cpp/releases)
(take both `llama-<tag>-bin-ubuntu-cuda-12.8-x64.tar.gz` **and**
`cudart-llama-<tag>-bin-ubuntu-cuda-12.8-x64.tar.gz`, and unpack the CUDA `.so` files into the
same directory as the binary — the official builds carry an `$ORIGIN` rpath).

## How it was built

Inside `nvidia/cuda:13.0.2-devel-ubuntu22.04` via Apptainer. CUDA 13.0 matches Colab's toolkit;
Ubuntu 22.04 pins glibc low enough to stay portable.

```bash
cmake -B build -S llama.cpp -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75-real \
  -DGGML_CUDA_CCCL_VERSION=v3.4.3 \
  -DGGML_CUDA_NCCL=OFF \
  -DGGML_STATIC=ON \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_BACKEND_DL=OFF \
  -DGGML_NATIVE=OFF \
  -DGGML_OPENMP=OFF \
  -DLLAMA_OPENSSL=OFF \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_APP=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DCMAKE_EXE_LINKER_FLAGS="-static-libstdc++ -static-libgcc"
cmake --build build -j --target llama-server
```

### One patch is required

`GGML_STATIC=ON` is what makes CMake select `libcudart_static.a` / `libcublas_static.a`, but
upstream also adds a full `-static` under the same flag
([`ggml/src/CMakeLists.txt`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/CMakeLists.txt)):

```cmake
if (GGML_STATIC)
    if (UNIX AND NOT APPLE)
        set(CMAKE_FIND_LIBRARY_SUFFIXES ".a;.so")   # wanted
    endif()
    add_link_options(-static)                        # fatal for CUDA
    if (MINGW)
        add_link_options(-static-libgcc -static-libstdc++)   # MinGW only!
    endif()
endif()
```

A full `-static` link cannot work here: `ggml-cuda` links `CUDA::cuda_driver`, and `libcuda` ships
only as a dynamic stub. Even if it linked, a fully static glibc binary cannot `dlopen` the driver.
Note also that the `-static-libgcc -static-libstdc++` line is guarded by `if (MINGW)` and never
runs on Linux — hence passing those flags explicitly via `CMAKE_EXE_LINKER_FLAGS`.

```bash
sed -i 's|^\([[:space:]]*\)add_link_options(-static)[[:space:]]*$|\1# patched out|' \
    ggml/src/CMakeLists.txt
```

## Verification

Every release is checked before upload:

1. `file` reports **dynamically linked** (a fully static result means the patch did not apply)
2. `ldd` lists only the four entries in the table above — no `libcudart`, `libcublas`, `libstdc++`
3. `cuobjdump --list-elf` contains `sm_75`
4. `readelf -V` shows max `GLIBC_2.34` and zero `GLIBCXX_` / `CXXABI_` symbols
5. the binary runs on a real GPU node and `--list-devices` enumerates the device

## License

The binary is a build of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), MIT licensed.
