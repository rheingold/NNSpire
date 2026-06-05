# NNSpire C++ Layer Plugin Scaffold

This scaffold generates a C++ shared library that implements a custom
NNSpire layer plugin.

## Prerequisites

- NNSpire SDK (source tree or installed headers)
- C++17 compiler (GCC 11+, Clang 14+, or MSVC 2022+)
- CMake 3.21+

## Getting started

1. **Rename placeholders** — edit all files and replace every `{{PLACEHOLDER}}`
   with your actual values (plugin ID, name, author, etc.).  Use the generator
   script for convenience:

   ```sh
   python generate_manifest.py \
       --id com.example.my-layer \
       --name "My Custom Layer" \
       --author "Example Corp" \
       --license MIT
   ```

2. **Generate signing key + CSR**:

   ```sh
   NNSpire-sign keygen --type layer --id com.example.my-layer --out .
   ```

3. **Build the plugin**:

   ```sh
   cmake -B build -DNNSpire_SDK_DIR=/path/to/NNSpire/plugin-api
   cmake --build build
   ```

4. **Compute sha256 and fill manifest**:

   ```sh
   python generate_manifest.py --fill-sha256 build/my_layer.dll
   ```

5. **Sign the manifest**:

   ```sh
   NNSpire-sign sign --manifest plugin.manifest.json \
       --cert plugin_signing.crt --key plugin_signing.key
   ```

6. **Install the plugin** — copy the `.dll`/`.so`, `plugin.manifest.json`,
   and `.p7s` to one of the NNSpire plugin directories:
   - `<user home>/.NNSpire/plugins/<your-plugin-id>/`

## Files

| File | Purpose |
|---|---|
| `CMakeLists.txt` | Build configuration |
| `include/my_layer.h` | Plugin class declaration |
| `src/my_layer.cpp` | Plugin implementation + vtable |
| `plugin.manifest.json.template` | Manifest template (filled by generate_manifest.py) |
| `generate_manifest.py` | Generates filled manifest + computes sha256 |

## Dual-language compliance

If you also provide a Python implementation of this layer, add a `python`
block to `implementations` in `plugin.manifest.json` and run the behavioural
equivalence check before releasing:

```sh
NNSpire-sign verify --manifest plugin.manifest.json --behavioral-check
```

@kb: PLUGIN-SDK.md §6
