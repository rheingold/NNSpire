# NNStudio Python Layer Plugin Scaffold

This scaffold generates a Python wheel that implements a custom NNStudio
layer plugin.

## Prerequisites

- Python 3.10+
- NNStudio Python bridge (`pip install nnstudio`)
- Hatchling: `pip install hatchling`

## Getting started

1. **Copy this directory** — replace `{{PLUGIN_ID_SNAKE}}` in directory
   names and all `{{PLACEHOLDER}}` tokens in file contents with your values.
   Use the manifest generator for convenience:

   ```sh
   python ../../generate_manifest.py \
       --id com.example.my-layer \
       --name "My Custom Layer" \
       --author "Example Corp" \
       --license MIT \
       --template plugin.manifest.json.template \
       --out plugin.manifest.json
   ```

2. **Implement the layer** — edit `{{PLUGIN_ID_SNAKE}}/layer.py` and fill in
   `forward()` and `backward()`.

3. **Build the wheel**:

   ```sh
   pip install hatchling
   python -m hatchling build
   ```

4. **Compute sha256 and fill manifest**:

   ```sh
   python ../../generate_manifest.py \
       --fill-sha256 dist/my_layer-1.0.0-py3-none-any.whl \
       --manifest plugin.manifest.json
   ```

5. **Sign the manifest**:

   ```sh
   nnstudio-sign sign --manifest plugin.manifest.json \
       --cert plugin_signing.crt --key plugin_signing.key
   ```

6. **Install** — copy the wheel, `plugin.manifest.json`, and `.p7s` to:
   `<user home>/.nnstudio/plugins/<your-plugin-id>/`
   Then `pip install` the wheel in the NNStudio Python environment.

## Files

| File | Purpose |
|---|---|
| `pyproject.toml.template` | Build configuration template |
| `{{PLUGIN_ID_SNAKE}}/layer.py` | Layer implementation |
| `{{PLUGIN_ID_SNAKE}}/__init__.py` | Package entry point |
| `plugin.manifest.json.template` | Manifest template |

## Dual-language compliance

To provide a C++ equivalent of this layer (required for registry submission),
follow the C++ layer scaffold instead, or extend the `implementations`
block in your manifest with both `cpp` and `python` entries.

@kb: PLUGIN-SDK.md §7
