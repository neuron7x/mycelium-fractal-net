"""Allow running: python -m mycelium_fractal_net.core --manifest"""

import sys

if "--manifest" in sys.argv or "--json" in sys.argv:
    # Delegate to causal_validation CLI
    from mycelium_fractal_net.core import causal_validation as _cv

    _cv_module = sys.modules[_cv.__name__]
    # Re-run the CLI block
    if "--manifest" in sys.argv:
        from mycelium_fractal_net.core.rule_registry import print_manifest

        print_manifest()
    elif "--json" in sys.argv:
        import json

        from mycelium_fractal_net.core.rule_registry import manifest_dict

        sys.stdout.write(json.dumps(manifest_dict(), indent=2) + "\n")
else:
    sys.stderr.write(
        "Usage:\n"
        "  python -m mycelium_fractal_net.core.causal_validation --manifest\n"
        "  python -m mycelium_fractal_net.core.causal_validation --json\n"
    )
