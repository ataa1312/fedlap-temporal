import sys; sys.path.insert(0, "/Users/ata/Desktop/master-thesis-workspace/master-thesis-codes/codes/fedlap")
import torch, numpy as np
sys.argv = ["x","-c","config/uci_gru.yaml","--set","model.data_type=f+s",
            "spectral.update_mode=recompute","subgraph.num_subgraphs=1","wandb.mode=disabled"]
from parser import Parser
p=Parser(); cfg=p.load_config(p.parse_args())
import src; src.config=cfg
from registries import datasets
import src.datasets
import scipy.stats as sst

snaps = datasets["uci"](cfg); N = snaps[0].num_nodes; T = len(snaps)
def und(ei):
    e = ei.cpu().numpy()
    return {(min(a,b),max(a,b)) for a,b in zip(e[0],e[1]) if a!=b}
def auc(pos,neg):
    s=np.concatenate([pos,neg]); r=sst.rankdata(s); npos=len(pos)
    return (r[:npos].sum()-npos*(npos+1)/2)/(npos*len(neg))

rng=np.random.default_rng(7)
cumset=set(); res={}
def add(k,v): res.setdefault(k,[]).append(v)
for t in range(T-1):
    for pr in und(snaps[t].edge_index): cumset.add(pr)
    # exact sym-normalized Laplacian eigendecomposition (dense)
    A=np.zeros((N,N),dtype=np.float64)
    for a,b in cumset: A[a,b]=1; A[b,a]=1
    d=A.sum(1); dis=np.where(d>0,1/np.sqrt(d),0)
    Lsym=np.eye(N)-(dis[:,None]*A*dis[None,:])
    w,V=np.linalg.eigh(Lsym)
    # low-k nontrivial eigenvectors (skip near-zero component indicators)
    nz=w>1e-8
    K=300
    Vlow=V[:,nz][:,:K]; wlow=w[nz][:K]
    V50=V[:,nz][:,:50]
    fut=list(und(snaps[t+1].edge_index))
    if len(fut)<10: continue
    def negs(n):
        out=[]
        while len(out)<n:
            a,b=rng.integers(0,N,2)
            if a==b: continue
            k2=(min(a,b),max(a,b))
            if k2 in cumset: continue
            out.append((int(a),int(b)))
        return out
    cur=list(cumset); ci=rng.choice(len(cur),size=min(len(cur),2000),replace=False)
    cur_s=[cur[i] for i in ci]
    nf=negs(len(fut)); nc=negs(len(cur_s))
    def cosv(M,pairs):
        out=[]
        for a,b in pairs:
            qa,qb=M[a],M[b]; na,nb=np.linalg.norm(qa),np.linalg.norm(qb)
            out.append(float(qa@qb/(na*nb)) if na>0 and nb>0 else 0.0)
        return np.array(out)
    for name,M in [("exact300",Vlow),("exact50",V50)]:
        add(f"CUR_{name}",auc(cosv(M,cur_s),cosv(M,nc)))
        add(f"FUT_{name}",auc(cosv(M,fut),cosv(M,nf)))
import statistics as st2
print(f"=== probe3 uci ({len(res['CUR_exact300'])} snaps) — EXACT dense eigh, sym-normalized, cosine affinity ===")
for k in ["CUR_exact300","CUR_exact50","FUT_exact300","FUT_exact50"]:
    print(f"  {k:14s} AUC={st2.mean(res[k]):.3f}")
