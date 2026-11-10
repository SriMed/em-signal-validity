"""Bundle the frozen development collector into an uploadable Colab notebook."""
import argparse
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ['scripts/run_phase3_development.py', 'tests/test_phase3_development.py',
         'configs/phase3-development.json', 'docs/labeling-pilot.md', 'requirements.txt']


def notebook():
    payload = {name: base64.b64encode((ROOT / name).read_bytes()).decode() for name in FILES}
    setup = (
        'import base64, json, subprocess, sys\nfrom pathlib import Path\n'
        "ROOT = Path('/content/phase3')\n"
        'def run(command):\n'
        '    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n'
        '    for line in process.stdout:\n        print(line, end="", flush=True)\n'
        '    if process.wait():\n        raise subprocess.CalledProcessError(process.returncode, command)\n'
        f'payload = {payload!r}\n'
        'for name, encoded in payload.items():\n'
        '    path = ROOT / name\n    path.parent.mkdir(parents=True, exist_ok=True)\n'
        '    path.write_bytes(base64.b64decode(encoded))\n'
        "run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(ROOT / 'requirements.txt')])\n"
        "# Colab's preinstalled torchvision may not match the pinned torch; this text-only run does not use it.\n"
        "run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchvision'])\n"
    )
    smoke = (
        "run([sys.executable, '-m', 'unittest', 'discover', '-s', str(ROOT / 'tests'), "
        "'-p', 'test_phase3_development.py', '-v'])\n"
        "run([sys.executable, str(ROOT / 'scripts/run_phase3_development.py'), '--validate-only'])\n"
        "import torch\nassert torch.cuda.is_available(), 'Select a GPU runtime before collection'\n"
        "print(torch.cuda.get_device_name(0))\n"
    )
    collect = (
        "OUT = Path('/content/phase3-development-run')\n"
        "run([sys.executable, '-u', str(ROOT / 'scripts/run_phase3_development.py'), "
        "'--output-dir', str(OUT)])\n"
    )
    retain = (
        'import hashlib, shutil\nfrom google.colab import files\n'
        "archive = Path(shutil.make_archive(str(OUT), 'zip', root_dir=OUT))\n"
        "print(json.dumps(json.loads((OUT / 'checks.json').read_text()), indent=2))\n"
        "print('Archive:', archive.name, 'bytes:', archive.stat().st_size)\n"
        "print('SHA256:', hashlib.sha256(archive.read_bytes()).hexdigest())\n"
        'files.download(str(archive))\n'
    )
    cells = [dict(cell_type='markdown', metadata={}, source=[
        '# Phase 3: 40-reply development collection\n',
        'Use a T4 GPU runtime and Run all. This notebook installs the pinned dependencies, '
        'tests extraction on a tiny randomly initialized Qwen, then runs the pinned EM model. '
        'It creates a fresh output directory and downloads the complete ZIP. '
        'Keep the download before ending the runtime. No human judgments are included or supplied.\n'])]
    for source in [setup, smoke, collect, retain]:
        cells.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[],
                          source=source.splitlines(keepends=True)))
    return dict(nbformat=4, nbformat_minor=5, cells=cells,
                metadata=dict(accelerator='GPU', colab=dict(name='03_phase3_development.ipynb',
                    provenance=[], gpuType='T4'), kernelspec=dict(name='python3', display_name='Python 3')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(notebook(), stream, indent=2)
        stream.write('\n')
    print(args.output)


if __name__ == '__main__':
    main()
