"""Freeze calibration labels/rubrics BEFORE predictions, not optimization v2."""
import json
from pathlib import Path
import subprocess

from evals.semantic_dataset_v2 import load_dataset, sha

ROOT=Path('data/evals/semantic_answer/v2')


def freeze():
    if (ROOT/'validation_manifest.json').exists(): raise RuntimeError('Calibration already frozen')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip(): raise RuntimeError('Commit clean source/data before freezing calibration')
    load_dataset(ROOT)
    files={p.name:sha(p) for p in sorted(ROOT.iterdir()) if p.is_file()}
    code_paths=sorted(Path('src/evals').glob('*v2.py'))+sorted(Path('scripts/evals/agents').glob('*v2.py'))
    # Include frozen adapters actually imported by new evaluation code.
    code_paths += [Path('src/evals/semantic_answer_v1.py')]
    manifest={'status':'CALIBRATION_FROZEN_BEFORE_JUDGE; OPTIMIZATION_BENCHMARK_NOT_YET_FROZEN',
              'construction_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'files_sha256':files,'code_sha256':{str(p):sha(p) for p in code_paths},
              'policy':'No label/rubric/candidate tuning after observing validation. Any failed gate disables unattended full-benchmark judging for this candidate policy.'}
    (ROOT/'validation_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'validation_manifest_sha256':sha(ROOT/'validation_manifest.json'),'construction_sha':manifest['construction_sha']},indent=2))


if __name__=='__main__': freeze()
