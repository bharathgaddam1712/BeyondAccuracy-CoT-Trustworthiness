"""Original script paths delegate to the single corrected implementation."""
import argparse
import subprocess
import sys
from src.protocol import ROOT

def legacy_main(stage):
    parser=argparse.ArgumentParser(description="Corrected pipeline entry point; original script path retained")
    parser.add_argument("--config",default=str(ROOT/"configs/blackwell_full.json"))
    parser.add_argument("--gpus",nargs="+",type=int,default=[0,1])
    parser.add_argument("--cpu-threads",type=int,default=8)
    args=parser.parse_args()
    command=[sys.executable,str(ROOT/"run_pipeline.py"),"analyze" if stage=="analyze" else "run","--config",args.config]
    if stage!="analyze":command.extend(["--gpus",*[str(i) for i in args.gpus],"--cpu-threads",str(args.cpu_threads),"--stages",stage])
    raise SystemExit(subprocess.call(command,cwd=ROOT))
