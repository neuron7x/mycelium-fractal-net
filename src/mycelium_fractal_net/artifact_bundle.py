from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _get_crypto():
    from importlib import import_module
    sig_mod = import_module('mycelium_fractal_net.crypto.signatures')
    return (
        sig_mod,
        sig_mod.SignatureKeyPair,
        sig_mod.sign_message,
        sig_mod.verify_signature,
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _derive_keypair_from_seed(seed_text: str):
    signature_impl, SignatureKeyPair, _, _ = _get_crypto()
    seed = hashlib.sha256(seed_text.encode("utf-8")).digest()
    h = signature_impl._sha512(seed)
    h_bytes = list(h[:32])
    h_bytes[0] &= 248
    h_bytes[31] &= 127
    h_bytes[31] |= 64
    a = int.from_bytes(bytes(h_bytes), "little")
    base = signature_impl._get_base_point()
    public_key = signature_impl._point_to_bytes(signature_impl._scalar_mult(a, base))
    return SignatureKeyPair(private_key=seed, public_key=public_key)


def _crypto_config_seed(config_path: str | Path) -> str:
    text = Path(config_path).read_text(encoding="utf-8")
    match = re.search(r'deterministic_artifact_seed:\s*"?([^"\n]+)"?', text)
    if match:
        return match.group(1).strip()
    return "mfn-artifact-signing-v1"


def _append_audit_event(audit_log: Path, event: dict[str, Any]) -> None:
    audit_log.parent.mkdir(parents=True, exist_ok=True)
    with audit_log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, sort_keys=True) + "\n")


def sign_artifact(
    path: str | Path,
    *,
    config_path: str | Path,
    audit_log: str | Path | None = None,
) -> Path:
    artifact_path = Path(path)
    _, _, sign_message_fn, _ = _get_crypto()
    keypair = _derive_keypair_from_seed(_crypto_config_seed(config_path))
    digest = sha256_file(artifact_path)
    signature = sign_message_fn(digest.encode("utf-8"), keypair.private_key)
    payload = {
        "schema_version": "mfn-artifact-signature-v1",
        "algorithm": "Ed25519",
        "path": artifact_path.name,
        "sha256": digest,
        "signature_hex": signature.hex(),
        "public_key_hex": keypair.public_key.hex(),
        "signed_at": datetime.now(timezone.utc).isoformat(),
    }
    sig_path = artifact_path.with_suffix(artifact_path.suffix + ".sig.json")
    sig_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if audit_log is not None:
        _append_audit_event(
            Path(audit_log),
            {
                "event": "sign",
                "path": str(artifact_path),
                "sha256": digest,
                "signature_path": str(sig_path),
                "timestamp": payload["signed_at"],
            },
        )
    return sig_path


def verify_artifact_signature(
    path: str | Path,
    *,
    signature_path: str | Path | None = None,
    audit_log: str | Path | None = None,
) -> bool:
    artifact_path = Path(path)
    sig_path = Path(signature_path) if signature_path is not None else artifact_path.with_suffix(artifact_path.suffix + ".sig.json")
    payload = json.loads(sig_path.read_text(encoding="utf-8"))
    _, _, _, verify_signature_fn = _get_crypto()
    digest = sha256_file(artifact_path)
    ok = digest == payload["sha256"] and verify_signature_fn(
        digest.encode("utf-8"),
        bytes.fromhex(payload["signature_hex"]),
        bytes.fromhex(payload["public_key_hex"]),
    )
    if audit_log is not None:
        _append_audit_event(
            Path(audit_log),
            {
                "event": "verify",
                "path": str(artifact_path),
                "signature_path": str(sig_path),
                "sha256": digest,
                "ok": bool(ok),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
    return bool(ok)


def sign_artifacts(
    paths: Iterable[str | Path], *, config_path: str | Path, audit_log: str | Path | None = None
) -> list[dict[str, str]]:
    signed = []
    for path in paths:
        sig_path = sign_artifact(path, config_path=config_path, audit_log=audit_log)
        signed.append({"artifact": Path(path).name, "signature": sig_path.name})
    return signed


def _verify_default_signature_if_present(target: Path, failures: list[str]) -> None:
    sig_path = target.with_suffix(target.suffix + ".sig.json")
    if not sig_path.exists():
        return
    if not verify_artifact_signature(target, signature_path=sig_path):
        failures.append(f"signature-invalid:{target.name}")


def manifest_entries_ok(manifest_path: str | Path) -> tuple[bool, list[str]]:
    path = Path(manifest_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    failures: list[str] = []
    if path.name == "manifest.json":
        for section in ("artifact_manifest", "optional_artifact_manifest"):
            for entry in data.get(section, {}).values() if isinstance(data.get(section), dict) else data.get(section, []):
                rel = entry["path"]
                target = path.parent / rel
                if not target.exists():
                    failures.append(f"missing:{rel}")
                    continue
                actual = sha256_file(target)
                if actual != entry["sha256"]:
                    failures.append(f"sha256-mismatch:{rel}")
                _verify_default_signature_if_present(target, failures)
    else:
        for entry in data.get("bundle_artifacts", []):
            rel = entry["path"]
            target = path.parent / rel
            if not target.exists():
                failures.append(f"missing:{rel}")
                continue
            actual = sha256_file(target)
            if actual != entry["sha256"]:
                failures.append(f"sha256-mismatch:{rel}")
            _verify_default_signature_if_present(target, failures)
    _verify_default_signature_if_present(path, failures)
    for entry in data.get("signatures", []):
        artifact = path.parent / entry["artifact"]
        signature_path = path.parent / entry["signature"]
        if not artifact.exists() or not signature_path.exists():
            failures.append(f"missing-signature:{entry}")
            continue
        if not verify_artifact_signature(artifact, signature_path=signature_path):
            failures.append(f"signature-invalid:{entry['artifact']}")
    return len(failures) == 0, failures


def verify_bundle(root_or_manifest: str | Path) -> dict[str, Any]:
    root = Path(root_or_manifest)
    manifests: list[Path]
    if root.is_file():
        manifests = [root]
    else:
        candidates = [root / "manifest.json", root / "release_manifest.json", root / "showcase_manifest.json"]
        manifests = [item for item in candidates if item.exists()]
        if not manifests:
            manifests = list(root.rglob("manifest.json")) + list(root.rglob("release_manifest.json")) + list(root.rglob("showcase_manifest.json"))
    results = []
    ok = True
    for manifest in manifests:
        manifest_ok, failures = manifest_entries_ok(manifest)
        ok = ok and manifest_ok
        results.append({"manifest": str(manifest), "ok": manifest_ok, "failures": failures})
    return {"ok": ok, "manifests": results}


__all__ = ["sha256_file", "sign_artifact", "verify_artifact_signature", "sign_artifacts", "verify_bundle"]
