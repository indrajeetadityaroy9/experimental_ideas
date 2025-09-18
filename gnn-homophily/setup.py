import argparse
import os
import subprocess
import sys
import textwrap
import importlib.util
import pathlib
import re
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description="Provision the OpenHGNN research environment.")
    parser.add_argument(
        "--kill-after",
        action="store_true",
        help="Forcefully terminate the interpreter with SIGKILL after setup (legacy behaviour).",
    )
    return parser.parse_args()


def run(cmd):
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    args = parse_args()
    python_executable = sys.executable

    run([python_executable, "-m", "pip", "uninstall", "-y",
         "torch", "torchvision", "torchaudio", "torch-geometric",
         "torch_scatter", "torch_sparse", "torch_cluster", "torch_spline_conv",
         "dgl", "openhgnn", "timm", "fastai"])

    run([python_executable, "-m", "pip", "cache", "purge"])

    idx = "https://download.pytorch.org/whl/cu121"
    run([python_executable, "-m", "pip", "install", f"--index-url={idx}", "torch==2.4.0", "torchvision==0.19.0",
         "torchaudio==2.4.0"])

    pyg_idx = "https://data.pyg.org/whl/torch-2.4.0+cu121.html"
    run([python_executable, "-m", "pip", "install",
         "torch_scatter==2.1.2", "torch_sparse==0.6.18",
         "torch_cluster==1.6.3", "torch_spline_conv==1.2.2",
         "-f", pyg_idx])

    run([python_executable, "-m", "pip", "install", "torch-geometric==2.6.1"])

    try:
        run([python_executable, "-m", "pip", "install", "dgl==2.4.0", "-f",
             "https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html"])
    except subprocess.CalledProcessError:
        run([python_executable, "-m", "pip", "install", "dgl==2.4.0"])

    run([python_executable, "-m", "pip", "install", "--no-deps", "git+https://github.com/BUPT-GAMMA/OpenHGNN@main"])

    run([python_executable, "-m", "pip", "install",
         "python-igraph==0.11.6", "texttable>=1.6.7",
         "lmdb", "rdflib==7.0.0", "ordered-set", "pyarrow",
         "ogb>=1.3.6", "timm==1.0.19", "fastai==2.8.3"])

    run([python_executable, "-m", "pip", "install", "networkx>=3.2"])

    spec = importlib.util.find_spec("openhgnn")
    assert spec and spec.origin, "OpenHGNN not found after install."
    pkg = pathlib.Path(spec.origin).parent
    tf_dir = pkg / "trainerflow"
    tf_dir.mkdir(parents=True, exist_ok=True)

    stub = tf_dir / "Ingram_trainer.py"
    if not stub.exists():
        stub.write_text(textwrap.dedent("""
            class Ingram_trainer:
                def __init__(self, *args, **kwargs):
                    raise NotImplementedError("Ingram_trainer is not available in this OpenHGNN build.")
        """).strip() + "\n", encoding="utf-8")

    pattern = r"from\s+networkx\.algorithms\.centrality\.betweenness\s+import\s+edge_betweenness\b"
    replacement = "from networkx.algorithms.centrality.betweenness import edge_betweenness_centrality as edge_betweenness"
    for py in tf_dir.glob("*.py"):
        src = py.read_text(encoding="utf-8")
        patched = re.sub(pattern, replacement, src)
        if patched != src:
            py.write_text(patched, encoding="utf-8")

    for pc in pkg.rglob("__pycache__"):
        shutil.rmtree(pc, ignore_errors=True)

    if args.kill_after:
        os.kill(os.getpid(), 9)
    else:
        print("Setup completed successfully!")


if __name__ == "__main__":
    main()
