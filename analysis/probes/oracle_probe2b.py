import os, sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT)); os.chdir(_ROOT)
import math, torch, numpy as np
sys.argv = ["x","-c","config/uci_gru.yaml","--set","model.data_type=f+s",
            "spectral.update_mode=recompute","subgraph.num_subgraphs=1","wandb.mode=disabled"]
from parser import Parser
p=Parser(); cfg=p.load_config(p.parse_args())
import src; src.config=cfg
from registries import datasets
import src.datasets
from src.utils.graph import Graph
import scipy.stats as sst

snaps = datasets["uci"](cfg); N = snaps[0].num_nodes; T = len(snaps)
def und(ei):
    e = ei.cpu().numpy()
    return {(min(a,b),max(a,b)) for a,b in zip(e[0],e[1]) if a!=b}
def auc(pos,neg):
    s=np.concatenate([pos,neg]); r=sst.rankdata(s); npos=len(pos)
    return (r[:npos].sum()-npos*(npos+1)/2)/(npos*len(neg))

rng = np.random.default_rng(7)
adj=[set() for _ in range(N)]; cum=None
res={k:[] for k in ["FUT_cos","FUT_heat1","FUT_heat10","FUT_dot","CUR_cos","CUR_heat1","CUR_heat10","CUR_dot"]}
for t in range(T-1):
    e=snaps[t].edge_index.cpu(); e2=torch.cat([e,e.flip(0)],dim=1)
    cum=e2 if cum is None else torch.unique(torch.cat([cum,e2],dim=1),dim=1)
    for a,b in und(e): adj[a].add(b); adj[b].add(a)
    g=Graph(x=torch.ones(N,1),edge_index=cum,node_ids=torch.arange(N))
    D,U,_=g.calc_eignvalues(estimate=True,spectral_len=300,log=False)
    Q=U.detach().float().cpu().numpy(); Dv=D.detach().float().cpu().numpy()
    w1=np.exp(-1.0*np.abs(Dv)); w10=np.exp(-10.0*np.abs(Dv))
    cur=set()
    ce=cum.numpy()
    for a,b in zip(ce[0],ce[1]):
        if a<b: cur.add((int(a),int(b)))
    fut=list(und(snaps[t+1].edge_index)); cur_l=list(cur)
    if len(fut)<10: continue
    def negs(pos,n):
        posset=set(pos); out=[]
        while len(out)<n:
            a,b=rng.integers(0,N,2)
            if a==b: continue
            k=(min(a,b),max(a,b))
            if k in posset or k in cur: continue
            out.append((int(a),int(b)))
        return out
    def sc(pairs):
        c,h1,h10,dt=[],[],[],[]
        for a,b in pairs:
            qa,qb=Q[a],Q[b]; na,nb=np.linalg.norm(qa),np.linalg.norm(qb)
            c.append(float(qa@qb/(na*nb)) if na>0 and nb>0 else 0.0)
            h1.append(float((qa*w1)@qb)); h10.append(float((qa*w10)@qb)); dt.append(float(qa@qb))
        return map(np.array,(c,h1,h10,dt))
    nf=negs(fut,len(fut)); nc=negs([],min(len(cur_l),2000))
    cur_s=rng.choice(len(cur_l),size=min(len(cur_l),2000),replace=False)
    cur_sample=[cur_l[i] for i in cur_s]
    Pf=sc(fut); Nf=sc(nf); Pc=sc(cur_sample); Nc=sc(nc)
    for i,k in enumerate(["cos","heat1","heat10","dot"]):
        res[f"FUT_{k}"].append(auc(list(Pf)[0] if False else None,None) if False else None)
    # redo cleanly
    Pf=list(sc(fut)); Nf=list(sc(nf)); Pc=list(sc(cur_sample)); Nc=list(sc(nc))
    for i,k in enumerate(["cos","heat1","heat10","dot"]):
        res[f"FUT_{k}"].append(auc(Pf[i],Nf[i])); res[f"CUR_{k}"].append(auc(Pc[i],Nc[i]))
import statistics as st2
print("=== probe2b uci — AUC binned EARLY / MID / LATE thirds ===")
for k in ["CUR_cos","CUR_dot","FUT_cos","FUT_dot"]:
    v=[x for x in res[k] if x is not None]
    n=len(v); a,b=n//3,2*n//3
    print(f"  {k:10s} early={st2.mean(v[:a]):.3f}  mid={st2.mean(v[a:b]):.3f}  late={st2.mean(v[b:]):.3f}  (n={n})")
