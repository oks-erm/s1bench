"""One launch command; creates a private environment on first use."""
from pathlib import Path
import argparse
import hashlib
import subprocess
import sys
import venv

BASE = Path(__file__).resolve().parent

def main():
    if sys.version_info < (3,10):
        raise SystemExit("Install Python 3.10 or newer.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",default="127.0.0.1")
    parser.add_argument("--port",type=int,default=8501)
    args = parser.parse_args()
    environment = BASE/".venv"
    python = environment/("Scripts/python.exe" if sys.platform=="win32" else "bin/python")
    if not python.exists():
        print("Creating the benchmark's private Python environment...",flush=True)
        venv.EnvBuilder(with_pip=True).create(environment)
    requirements = BASE/"requirements.txt"
    digest = hashlib.sha256(requirements.read_bytes()).hexdigest()
    marker = environment/".benchmark_dependencies"
    installed = marker.exists() and marker.read_text().strip()==digest
    probe = subprocess.run([str(python),"-c","import streamlit, pandas, plotly"],
                           stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    if not installed or probe.returncode:
        print("Installing dashboard dependencies...",flush=True)
        subprocess.run([str(python),"-m","pip","install","-r",str(requirements)],cwd=BASE,check=True)
        marker.write_text(digest)
    print("Starting dashboard.",flush=True)
    try:
        return subprocess.call([str(python),"-m","streamlit","run",str(BASE/"bench_ui.py"),
                                "--server.address",args.host,"--server.port",str(args.port)],cwd=BASE)
    except KeyboardInterrupt:
        return 0

if __name__=="__main__":
    raise SystemExit(main())
