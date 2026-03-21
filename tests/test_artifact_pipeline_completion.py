from __future__ import annotations
import json
import mycelium_fractal_net as mfn

def test_manifest_and_required_artifacts(tmp_path) -> None:
    seq = mfn.simulate(mfn.SimulationSpec(grid_size=16, steps=8, seed=42))
    report = mfn.report(seq, output_root=str(tmp_path), horizon=4)
    run_dir = tmp_path / report.run_id
    core_artifacts = {'config.json', 'field.npy', 'history.npy', 'descriptor.json', 'detection.json', 'forecast.json', 'comparison.json', 'report.md'}
    on_disk = {p.name for p in run_dir.iterdir()}
    assert core_artifacts.issubset(on_disk)
    assert 'manifest.json' in on_disk
    manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['run_id'] == report.run_id
    assert core_artifacts == set(manifest['artifact_list'])
