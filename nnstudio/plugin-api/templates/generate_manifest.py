#!/usr/bin/env python3
"""
generate_manifest.py — NNStudio Plugin Manifest Generator

Fills plugin.manifest.json.template with concrete values and optionally
computes SHA-256 hashes of build artefacts.

USAGE
─────
    # Fill template with metadata
    python generate_manifest.py \\
        --id com.example.my-layer \\
        --name "My Custom Layer" \\
        --version 1.0.0 \\
        --author "Example Corp" \\
        --author-url https://example.com \\
        --license MIT \\
        --description "A custom activation layer" \\
        --template plugin.manifest.json.template \\
        --out plugin.manifest.json

    # Fill sha256 for a C++ build artefact
    python generate_manifest.py \\
        --fill-sha256 build/my_layer.dll \\
        --manifest plugin.manifest.json

    # Fill sha256 for a Python wheel
    python generate_manifest.py \\
        --fill-sha256 dist/my_layer-1.0.0-py3-none-any.whl \\
        --manifest plugin.manifest.json

@kb: PLUGIN-SDK.md §8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys


def sha256_file(path: str) -> str:
    """Compute hex SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def to_snake(s: str) -> str:
    """Convert reverse-domain id to snake_case module name."""
    # com.example.my-layer → com_example_my_layer
    return re.sub(r"[.-]", "_", s)


def fill_template(template_text: str, values: dict) -> str:
    """Replace {{KEY}} placeholders in template_text."""
    result = template_text
    for key, val in values.items():
        result = result.replace("{{" + key + "}}", val)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate or update an NNStudio plugin.manifest.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Template fill options
    parser.add_argument("--id",          help="Reverse-domain plugin ID (e.g. com.example.my-layer)")
    parser.add_argument("--name",        help="Human-readable plugin name")
    parser.add_argument("--version",     default="1.0.0", help="Plugin version (semver, default: 1.0.0)")
    parser.add_argument("--author",      help="Author name")
    parser.add_argument("--author-url",  default="", help="Author URL (optional)")
    parser.add_argument("--license",     default="MIT", help="SPDX license identifier (default: MIT)")
    parser.add_argument("--description", default="", help="Short plugin description")
    parser.add_argument("--template",    default="plugin.manifest.json.template",
                        help="Path to manifest template (default: plugin.manifest.json.template)")
    parser.add_argument("--out",         default="plugin.manifest.json",
                        help="Output path for filled manifest (default: plugin.manifest.json)")

    # SHA-256 fill option
    parser.add_argument("--fill-sha256", metavar="ARTEFACT_PATH",
                        help="Compute SHA-256 of the given artefact and update the manifest in-place")
    parser.add_argument("--manifest",    default="plugin.manifest.json",
                        help="Manifest file to update when using --fill-sha256")

    args = parser.parse_args()

    # ── Mode: fill sha256 ────────────────────────────────────────────────────
    if args.fill_sha256:
        artefact = args.fill_sha256
        if not os.path.exists(artefact):
            print(f"error: artefact not found: {artefact}", file=sys.stderr)
            sys.exit(1)

        digest = sha256_file(artefact)
        print(f"SHA-256({os.path.basename(artefact)}) = {digest}")

        with open(args.manifest, encoding="utf-8") as fh:
            manifest = json.load(fh)

        impls = manifest.get("implementations", {})

        # Detect cpp or python based on the file extension
        ext = os.path.splitext(artefact)[1].lower()
        if ext in (".dll", ".so", ".dylib"):
            if "cpp" in impls:
                impls["cpp"]["sha256"] = digest
                impls["cpp"]["library"] = os.path.basename(artefact)
                print(f"Updated implementations.cpp.sha256 in {args.manifest}")
            else:
                print(f"warning: no 'cpp' entry in implementations; sha256 not written",
                      file=sys.stderr)
        elif ext == ".whl":
            if "python" in impls:
                impls["python"]["sha256"] = digest
                impls["python"]["wheel"] = os.path.basename(artefact)
                print(f"Updated implementations.python.sha256 in {args.manifest}")
            else:
                print(f"warning: no 'python' entry in implementations; sha256 not written",
                      file=sys.stderr)
        else:
            # Assume either cpp or python based on presence
            found = False
            for lang in ("cpp", "python"):
                if lang in impls:
                    impls[lang]["sha256"] = digest
                    print(f"Updated implementations.{lang}.sha256 in {args.manifest}")
                    found = True
                    break
            if not found:
                print(f"warning: no implementation entry found; sha256 not written",
                      file=sys.stderr)

        with open(args.manifest, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
            fh.write("\n")
        return

    # ── Mode: fill template ──────────────────────────────────────────────────
    if not args.id:
        parser.error("--id is required when filling a template (not using --fill-sha256)")

    if not os.path.exists(args.template):
        print(f"error: template not found: {args.template}", file=sys.stderr)
        sys.exit(1)

    with open(args.template, encoding="utf-8") as fh:
        template_text = fh.read()

    plugin_id    = args.id
    plugin_snake = to_snake(plugin_id)
    # Derive a CamelCase class name from the last segment of the id
    last_segment = plugin_id.rsplit(".", 1)[-1]
    class_name   = "".join(word.capitalize() for word in re.split(r"[-_]", last_segment))

    values = {
        "PLUGIN_REVERSE_DOMAIN_ID": plugin_id,
        "PLUGIN_ID_SNAKE":          plugin_snake,
        "PLUGIN_CLASS_NAME":        class_name,
        "PLUGIN_NAME":              args.name or class_name,
        "PLUGIN_VERSION":           args.version,
        "PLUGIN_AUTHOR":            args.author or "Unknown",
        "PLUGIN_AUTHOR_URL":        args.author_url,
        "PLUGIN_LICENSE":           args.license,
        "PLUGIN_DESCRIPTION":       args.description,
        "SHA256_CPP":               "<compute with --fill-sha256>",
        "SHA256_PYTHON":            "<compute with --fill-sha256>",
    }

    filled = fill_template(template_text, values)

    # Validate the result parses as JSON
    try:
        json.loads(filled)
    except json.JSONDecodeError as exc:
        print(f"warning: generated manifest has JSON errors: {exc}", file=sys.stderr)

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(filled)

    print(f"Written: {args.out}")
    print(f"  id:      {plugin_id}")
    print(f"  name:    {values['PLUGIN_NAME']}")
    print(f"  version: {args.version}")
    print(f"  class:   {class_name}")
    print()
    print("Next steps:")
    print("  1. Build your plugin artefact (cmake --build or pip wheel)")
    print("  2. python generate_manifest.py --fill-sha256 <artefact> --manifest", args.out)
    print("  3. nnstudio-sign sign --manifest", args.out, "--cert plugin.crt --key plugin.key")


if __name__ == "__main__":
    main()
