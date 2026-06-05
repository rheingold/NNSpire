"""
NNSpire.plugin_manifest — Plugin and TUP manifest parsing / validation.

Validates plugin.manifest.json and tup.manifest.json files against the same
rules as the C++ ManifestLoader.  Optionally uses the ``jsonschema`` package
(``pip install jsonschema``) for full JSON Schema Draft 7 validation; falls
back to manual field checking when ``jsonschema`` is not installed.

USAGE
─────
    from NNSpire.plugin_manifest import load_plugin_manifest, load_tup_manifest

    # Validate and parse a plugin manifest
    result = load_plugin_manifest("path/to/plugin.manifest.json")
    if result.ok:
        m = result.value
        print(m.id, m.version, m.type)
    else:
        print("Invalid manifest:", result.error)

    # Validate and parse a TUP manifest
    result = load_tup_manifest("path/to/tup.manifest.json")

@kb: PLUGIN-SDK.md §4  |  TRUST-ARCHITECTURE.md §TUP
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

# ─── Optional jsonschema ────────────────────────────────────────────────────

try:
    import jsonschema  # type: ignore
    _HAS_JSONSCHEMA = True
except ImportError:
    _HAS_JSONSCHEMA = False

# Path to the schema files relative to this module
_PKG_DIR    = os.path.dirname(os.path.abspath(__file__))
_SCHEMA_DIR = os.path.normpath(
    os.path.join(_PKG_DIR, "..", "plugin-api", "schemas")
)

# ─── Result type ────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Simple Result monad for manifest validation."""
    ok: bool
    value: object = None
    error: str = ""

    @staticmethod
    def success(value: object) -> "ValidationResult":
        return ValidationResult(ok=True, value=value)

    @staticmethod
    def failure(message: str) -> "ValidationResult":
        return ValidationResult(ok=False, error=message)


# ─── Data classes ────────────────────────────────────────────────────────────

@dataclass
class PluginImplementation:
    language: str           # "cpp" or "python"
    artifact: str           # library path or module name
    sha256: str             # hex SHA-256
    wheel: str = ""         # Python wheel path (optional)


@dataclass
class PluginManifest:
    manifest_version: int
    id: str
    name: str
    version: str
    type: str               # e.g. "LAYER", "TOKENIZER"
    author: str
    license: str
    min_studio_version: str
    capabilities: List[str]
    implementations: List[PluginImplementation]
    signature: str
    max_studio_version: str = ""
    description: str = ""
    kb_ref: str = ""
    author_url: str = ""
    license_url: str = ""


@dataclass
class TupCertEntry:
    file: str
    sha256: str
    role: str               # "root", "intermediate", or "crl"
    subject: str = ""


@dataclass
class TupRevocation:
    serial_number: str
    issuer_subject: str
    reason: str
    comment: str = ""


@dataclass
class TupManifest:
    manifest_version: int
    id: str                 # UUID v4
    version: str
    issued_at: str          # ISO 8601
    issuer_subject: str
    add_certs: List[TupCertEntry]
    revoke_certs: List[TupRevocation]
    signature: str
    description: str = ""
    self_revocation_guard: bool = True
    require_user_confirmation: bool = True


# ─── Patterns ────────────────────────────────────────────────────────────────

_SEMVER_RE   = re.compile(r"^\d+\.\d+\.\d+$")
_SHA256_RE   = re.compile(r"^[0-9a-f]{64}$")
_UUID4_RE    = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
_RDOMAIN_RE  = re.compile(r"^[a-z][a-z0-9]*(\.[a-z][a-z0-9-]*)+$")

_VALID_TYPES = {
    "LAYER", "ACTIVATION", "OPTIMIZER", "TOKENIZER", "BACKEND",
    "INPUT_ADAPTER", "OUTPUT_ADAPTER", "CONTEXT_SOURCE", "RUNNER_CLIENT",
    "UI_PANEL",
}  # TRUST_UPDATE excluded — that path uses tup.manifest.json


# ─── Internal helpers ────────────────────────────────────────────────────────

def _require(d: dict, key: str, context: str = "") -> ValidationResult:
    prefix = f"{context}: " if context else ""
    if key not in d:
        return ValidationResult.failure(f"{prefix}missing required field '{key}'")
    val = d[key]
    if val is None or val == "" or val == [] or val == {}:
        return ValidationResult.failure(f"{prefix}field '{key}' must not be empty")
    return ValidationResult.success(val)


def _schema_validate(data: dict, schema_name: str) -> Optional[str]:
    """
    Run jsonschema validation if available.
    Returns an error string on failure, or None on success.
    """
    if not _HAS_JSONSCHEMA:
        return None
    schema_path = os.path.join(_SCHEMA_DIR, schema_name)
    if not os.path.exists(schema_path):
        return None
    with open(schema_path, encoding="utf-8") as fh:
        schema = json.load(fh)
    try:
        jsonschema.validate(instance=data, schema=schema)
        return None
    except jsonschema.ValidationError as exc:
        return str(exc.message)


# ─── Plugin manifest loading ─────────────────────────────────────────────────

def _parse_implementations(raw: dict) -> ValidationResult:
    impls_raw = raw.get("implementations", {})
    if not impls_raw:
        return ValidationResult.failure("'implementations' must have at least one of 'cpp' or 'python'")

    impls: List[PluginImplementation] = []
    for lang in ("cpp", "python"):
        entry = impls_raw.get(lang)
        if not entry:
            continue
        if lang == "cpp":
            artifact = entry.get("library", "")
        else:
            artifact = entry.get("module", "")
        sha256 = entry.get("sha256", "")
        wheel  = entry.get("wheel", "")
        if not artifact:
            return ValidationResult.failure(
                f"implementations.{lang}: missing artifact field (library/module)"
            )
        if not sha256 or not _SHA256_RE.match(sha256):
            return ValidationResult.failure(
                f"implementations.{lang}: sha256 must be a 64-char lowercase hex string"
            )
        impls.append(PluginImplementation(language=lang, artifact=artifact,
                                           sha256=sha256, wheel=wheel))
    if not impls:
        return ValidationResult.failure(
            "'implementations' must have at least one of 'cpp' or 'python'"
        )
    return ValidationResult.success(impls)


def load_plugin_manifest(path_or_text: str) -> ValidationResult:
    """
    Parse and validate a plugin.manifest.json.

    `path_or_text` can be either:
    - A file-system path to a .json file, or
    - A raw JSON string (must start with ``{``).

    Returns a ``ValidationResult`` whose ``value`` is a ``PluginManifest``
    on success, or whose ``error`` describes the first violation.
    """
    if path_or_text.strip().startswith("{"):
        json_text = path_or_text
    else:
        try:
            with open(path_or_text, encoding="utf-8") as fh:
                json_text = fh.read()
        except OSError as exc:
            return ValidationResult.failure(f"cannot read manifest file: {exc}")

    try:
        raw = json.loads(json_text)
    except json.JSONDecodeError as exc:
        return ValidationResult.failure(f"invalid JSON: {exc}")

    # ── JSON Schema Draft 7 validation (if jsonschema is available) ──────────
    schema_err = _schema_validate(raw, "plugin.manifest.schema.json")
    if schema_err:
        return ValidationResult.failure(schema_err)

    # ── Manual required-field checks ────────────────────────────────────────
    ctx = "plugin.manifest"

    for req in ("manifestVersion", "id", "name", "version", "type",
                "author", "license", "minStudioVersion", "capabilities",
                "implementations", "signature"):
        rv = _require(raw, req, ctx)
        if not rv.ok:
            return rv

    if raw.get("manifestVersion") != 1:
        return ValidationResult.failure(f"{ctx}: 'manifestVersion' must be 1")

    plugin_type = raw["type"]
    if plugin_type == "TRUST_UPDATE":
        return ValidationResult.failure(
            f"{ctx}: type 'TRUST_UPDATE' is reserved for TUP packages"
        )
    if plugin_type not in _VALID_TYPES:
        return ValidationResult.failure(f"{ctx}: unknown plugin type '{plugin_type}'")

    if not _RDOMAIN_RE.match(raw["id"]):
        return ValidationResult.failure(
            f"{ctx}: 'id' must be a reverse-domain identifier, e.g. com.example.my-layer"
        )

    if not _SEMVER_RE.match(raw.get("version", "")):
        return ValidationResult.failure(
            f"{ctx}: 'version' must be an x.y.z semver string"
        )

    if not _SEMVER_RE.match(raw.get("minStudioVersion", "")):
        return ValidationResult.failure(
            f"{ctx}: 'minStudioVersion' must be an x.y.z semver string"
        )

    if not isinstance(raw.get("capabilities"), list) or not raw["capabilities"]:
        return ValidationResult.failure(
            f"{ctx}: 'capabilities' must be a non-empty array"
        )

    impls_rv = _parse_implementations(raw)
    if not impls_rv.ok:
        return impls_rv

    m = PluginManifest(
        manifest_version    = raw["manifestVersion"],
        id                  = raw["id"],
        name                = raw["name"],
        version             = raw["version"],
        type                = raw["type"],
        author              = raw["author"],
        license             = raw["license"],
        min_studio_version  = raw["minStudioVersion"],
        capabilities        = list(raw["capabilities"]),
        implementations     = impls_rv.value,
        signature           = raw["signature"],
        max_studio_version  = raw.get("maxStudioVersion", ""),
        description         = raw.get("description", ""),
        kb_ref              = raw.get("kbRef", ""),
        author_url          = raw.get("authorUrl", ""),
        license_url         = raw.get("licenseUrl", ""),
    )
    return ValidationResult.success(m)


# ─── TUP manifest loading ────────────────────────────────────────────────────

def load_tup_manifest(path_or_text: str) -> ValidationResult:
    """
    Parse and validate a tup.manifest.json.

    `path_or_text` can be a file-system path or a raw JSON string.

    Returns a ``ValidationResult`` whose ``value`` is a ``TupManifest``
    on success, or whose ``error`` describes the first violation.
    """
    if path_or_text.strip().startswith("{"):
        json_text = path_or_text
    else:
        try:
            with open(path_or_text, encoding="utf-8") as fh:
                json_text = fh.read()
        except OSError as exc:
            return ValidationResult.failure(f"cannot read TUP manifest: {exc}")

    try:
        raw = json.loads(json_text)
    except json.JSONDecodeError as exc:
        return ValidationResult.failure(f"invalid JSON: {exc}")

    # ── JSON Schema Draft 7 validation ───────────────────────────────────────
    schema_err = _schema_validate(raw, "tup.manifest.schema.json")
    if schema_err:
        return ValidationResult.failure(schema_err)

    # ── Manual required-field checks ────────────────────────────────────────
    ctx = "tup.manifest"

    for req in ("manifestVersion", "id", "type", "version",
                "issuedAt", "issuerSubject", "addCerts", "revokeCerts", "signature"):
        rv = _require(raw, req, ctx)
        if not rv.ok:
            # addCerts and revokeCerts may be empty lists — only check presence
            if req in ("addCerts", "revokeCerts") and req in raw:
                continue
            return rv

    if raw.get("manifestVersion") != 1:
        return ValidationResult.failure(f"{ctx}: 'manifestVersion' must be 1")

    if raw.get("type") != "TRUST_UPDATE":
        return ValidationResult.failure(f"{ctx}: 'type' must be 'TRUST_UPDATE'")

    if not _UUID4_RE.match(raw.get("id", "")):
        return ValidationResult.failure(f"{ctx}: 'id' must be a valid UUID v4")

    if not _SEMVER_RE.match(raw.get("version", "")):
        return ValidationResult.failure(
            f"{ctx}: 'version' must be an x.y.z semver string"
        )

    add_certs: List[TupCertEntry] = []
    for entry in raw.get("addCerts", []):
        file_   = entry.get("file", "")
        sha256_ = entry.get("sha256", "")
        role_   = entry.get("role", "")
        if not file_ or not sha256_ or not role_:
            return ValidationResult.failure(
                f"{ctx}: each addCerts entry must have 'file', 'sha256', and 'role'"
            )
        if not _SHA256_RE.match(sha256_):
            return ValidationResult.failure(
                f"{ctx}: addCerts entry '{file_}': sha256 must be 64 hex chars"
            )
        if role_ not in ("root", "intermediate", "crl"):
            return ValidationResult.failure(
                f"{ctx}: addCerts entry '{file_}': role must be 'root', 'intermediate', or 'crl'"
            )
        add_certs.append(TupCertEntry(
            file=file_, sha256=sha256_, role=role_,
            subject=entry.get("subject", "")
        ))

    revoke_certs: List[TupRevocation] = []
    valid_reasons = {
        "keyCompromise", "caCompromise", "affiliationChanged",
        "superseded", "cessationOfOperation", "unspecified"
    }
    for entry in raw.get("revokeCerts", []):
        sn     = entry.get("serialNumber", "")
        issuer = entry.get("issuerSubject", "")
        reason = entry.get("reason", "")
        if not sn or not issuer or not reason:
            return ValidationResult.failure(
                f"{ctx}: each revokeCerts entry must have 'serialNumber', 'issuerSubject', and 'reason'"
            )
        if reason not in valid_reasons:
            return ValidationResult.failure(
                f"{ctx}: revokeCerts entry: unknown reason '{reason}'"
            )
        revoke_certs.append(TupRevocation(
            serial_number=sn, issuer_subject=issuer,
            reason=reason, comment=entry.get("comment", "")
        ))

    t = TupManifest(
        manifest_version          = raw["manifestVersion"],
        id                        = raw["id"],
        version                   = raw["version"],
        issued_at                 = raw["issuedAt"],
        issuer_subject            = raw["issuerSubject"],
        add_certs                 = add_certs,
        revoke_certs              = revoke_certs,
        signature                 = raw["signature"],
        description               = raw.get("description", ""),
        self_revocation_guard     = raw.get("selfRevocationGuard", True),
        require_user_confirmation = raw.get("requireUserConfirmation", True),
    )
    return ValidationResult.success(t)
