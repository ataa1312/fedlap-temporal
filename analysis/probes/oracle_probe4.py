import sys; sys.path.insert(0, "/Users/ata/Desktop/master-thesis-workspace/master-thesis-codes/codes/fedlap")
import math, torch, numpy as np
sys.argv = ["x","-c","config/uci_gru.yaml","--set","model.data_type=f+s",
            "spectral.update_mode=recompute","subgraph.num_subgraphs=1","wandb.mode=disabled"]
from parser import Parser
p=Parser(); cfg=p.load_config(p.parse_args())
import src; src.config=cfg
from registries import datasets
import src.datasets
import scipy.stats as sst

snaps=datasets["uci"](cfg); N=snaps[0].num_nodes; T=len(snaps)
def und(ei):
    e=ei.cpu().numpy(); return {(min(a,b),max(a,b)) for a,b in zip(e[0],e[1]) if a!=b}
def auc(pos,neg):
    s=np.concatenate([pos,neg]); r=sst.rankdata(s); npos=len(pos)
    return (r[:npos].sum()-npos*(npos+1)/2)/(npos*len(neg))
rng=np.random.default_rng(7)
adj=[set() for _ in range(N)]; cumset=set(); res={}
def add(k,v): res.setdefault(k,[]).append(v)
for t in range(T-1):
    for a,b in und(snaps[t].edge_index):
        cumset.add((a,b)); adj[a].add(b); adj[b].add(a)
    A=np.zeros((N,N))
    for a,b in cumset: A[a,b]=1; A[b,a]=1
    d=A.sum(1); dis=np.where(d>0,1/np.sqrt(d),0)
    w,V=np.linalg.eigh(np.eye(N)-(dis[:,None]*A*dis[None,:]))
    V50=V[:,w>1e-8][:,:50]
    fut=list(und(snaps[t+1].edge_index))
    if len(fut)<10: continue
    negs=[]
    while len(negs)<len(fut):
        a,b=rng.integers(0,N,2)
        if a==b or (min(a,b),max(a,b)) in cumset: continue
        negs.append((int(a),int(b)))
    def feats(pairs):
        aa,sp=[],[]
        for a,b in pairs:
            aa.append(sum(1.0/math.log(max(len(adj[c]),2)) for c in adj[a]&adj[b]))
            qa,qb=V50[a],V50[b]; na,nb=np.linalg.norm(qa),np.linalg.norm(qb)
            sp.append(float(qa@qb/(na*nb)) if na>0 and nb>0 else 0.0)
        return np.array(aa),np.array(sp)
    Paa,Psp=feats(fut); Naa,Nsp=feats(negs)
    add("AA",auc(Paa,Naa)); add("SPEC_exact50",auc(Psp,Nsp))
    pm,nm=Paa==0,Naa==0
    if pm.sum()>=10 and nm.sum()>=10:
        add("SPEC_exact50@AAblind",auc(Psp[pm],Nsp[nm]))
        add("frac_pos_AAblind",float(pm.mean()))
import statistics as st2
print(f"=== probe4 uci ({len(res['AA'])} snaps) — is EXACT spectral signal complementary to local structure? ===")
for k,v in res.items(): print(f"  {k:22s} {st2.mean(v):.3f} (n={len(v)})")
