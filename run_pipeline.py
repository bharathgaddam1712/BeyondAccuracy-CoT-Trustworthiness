"""CLI for the corrected WLLFM pipeline. Run from any directory."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from src.protocol import ROOT,load_config,prepare,check_manifest,generate_paraphrases,atomic_json

def preflight(c,gpus):
    import torch
    if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable. Install the CUDA 12.8 PyTorch wheel")
    print(f"PyTorch {torch.__version__}; wheel CUDA {torch.version.cuda}")
    cards=[]
    for i in gpus:
        if i>=torch.cuda.device_count():raise ValueError(f"GPU {i} unavailable")
        torch.cuda.set_device(i);free,total=torch.cuda.mem_get_info(i);props=torch.cuda.get_device_properties(i)
        # Execute a kernel: catches unsupported GPU architecture instead of trusting version strings.
        x=torch.ones((16,16),device=f"cuda:{i}");float((x@x).sum().item());del x
        card=dict(index=i,name=props.name,capability=list(torch.cuda.get_device_capability(i)),free_gib=free/2**30,total_gib=total/2**30)
        cards.append(card);print(json.dumps(card))
        if free/2**30<24:raise RuntimeError(f"GPU {i} has less than 24 GiB free; wait for other jobs")
    root=prepare(c);atomic_json(root/"hardware.json",dict(torch=torch.__version__,cuda=torch.version.cuda,gpus=cards))
    return root

def assignments(c,gpus):
    from src.models import get_model_by_name
    buckets=[[] for _ in gpus];load=[0. for _ in gpus]
    for name in sorted(c["models"],key=lambda m:get_model_by_name(m).size_b,reverse=True):
        index=min(range(len(gpus)),key=lambda i:load[i]);buckets[index].append(name);load[index]+=get_model_by_name(name).size_b
    return buckets

def launch(c,args,stages):
    check_manifest(c);processes=[]
    try:
        for gpu,names in zip(args.gpus,assignments(c,args.gpus)):
            command=[sys.executable,str(Path(__file__).resolve()),"worker","--config",str(Path(args.config).resolve()),"--device",f"cuda:{gpu}","--models",*names,"--stages",*stages]
            env=dict(os.environ,OMP_NUM_THREADS=str(args.cpu_threads),TOKENIZERS_PARALLELISM="false",PYTHONUNBUFFERED="1")
            print("Starting",gpu,names,flush=True);processes.append(subprocess.Popen(command,cwd=ROOT,env=env))
        while processes:
            for p in list(processes):
                code=p.poll()
                if code is not None:
                    processes.remove(p)
                    if code:raise RuntimeError(f"GPU worker {p.pid} failed with exit {code}; raw checkpoints retained")
            if processes:
                import time
                time.sleep(1)
    finally:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:p.wait()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["preflight","prepare","paraphrases","run","all","analyze","validate","worker","plan","timing"])
    parser.add_argument("--config",default=str(ROOT/"configs/blackwell_full.json"))
    parser.add_argument("--gpus",type=int,nargs="+",default=[0,1])
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--models",nargs="+")
    parser.add_argument("--stages",nargs="+",choices=["behavioral","consistency","ablations","probes","quantization"],default=["behavioral","consistency","ablations","probes","quantization"])
    parser.add_argument("--cpu-threads",type=int,default=8)
    parser.add_argument("--require-human-review",action="store_true")
    args=parser.parse_args();c=load_config(args.config)
    if args.command=="timing":
        from src.protocol import read_jsonl
        root,_=check_manifest(c)
        from collections import defaultdict
        totals=defaultdict(lambda:dict(seconds=0.,n_responses=0,generated_tokens=0))
        for path in (root/"performance").glob("*.jsonl"):
            for r in read_jsonl(path):
                key=path.stem+"/"+r["stage"]
                for k in totals[key]:totals[key][k]+=r.get(k,0)
        for key,value in sorted(totals.items()):
            value["generated_tokens_per_second"]=value["generated_tokens"]/value["seconds"] if value["seconds"] else None
            print(key,json.dumps(value))
        return
    if args.command=="plan":
        print(json.dumps(dict(config=c,gpu_assignments=assignments(c,args.gpus)),indent=2));return
    if args.command=="prepare":prepare(c)
    elif args.command=="preflight":preflight(c,args.gpus)
    elif args.command=="paraphrases":generate_paraphrases(c,args.device)
    elif args.command=="worker":
        from src.runner import run_worker
        run_worker(c,args.models,args.device,args.stages)
    elif args.command in ("run","all"):
        if args.command=="all":
            preflight(c,args.gpus);generate_paraphrases(c,f"cuda:{args.gpus[0]}")
        launch(c,args,args.stages)
        from src.analysis import analyze
        analyze(c,require_complete=set(args.stages)=={"behavioral","consistency","ablations","probes","quantization"},require_human=args.require_human_review)
    elif args.command in ("analyze","validate"):
        from src.analysis import analyze,load_records,validate_outputs
        if args.command=="analyze":analyze(c,require_human=args.require_human_review)
        else:
            root,_=check_manifest(c);report=validate_outputs(c,root,load_records(root),args.require_human_review);print(json.dumps(report,indent=2))
            if report["errors"]:sys.exit(1)

if __name__=="__main__":main()
