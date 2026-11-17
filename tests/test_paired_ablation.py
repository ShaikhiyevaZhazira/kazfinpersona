import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import numpy as np
from paired_ablation import *

def rec(i, gold_a, gold_p, pa, pp, cf_of=None):
    return dict(scenario_id=i, gold_action=gold_a, gold_product_id=gold_p, pred_action=pa, pred_product_id=pp, counterfactual_of=cf_of)

# McNemar: 10 discordant, 9 vs 1 -> p = 2*P(X<=1 | n=10) = 0.0215
a={f'i{k}':0.0 for k in range(20)}; b={f'i{k}':0.0 for k in range(20)}
for k in range(9): b[f'i{k}']=1.0
a['i9']=1.0
for k in range(10,14): a[f'i{k}']=b[f'i{k}']=1.0
m=mcnemar_exact(a,b); print(m); assert m['only_a_correct']==1 and m['only_b_correct']==9 and abs(m['p_exact']-0.021484375)<1e-9
assert mcnemar_exact({'x':1.0},{'x':1.0})['p_exact']==1.0
# pma vector excludes CF and non-recommend gold
r=[rec('s1','recommend','p','recommend','p'),rec('s2','refuse',None,'recommend','p'),rec('s1-CF-x','recommend','q','recommend','q','s1')]
assert pma_vector(r)=={'s1':1.0}
# pair indicators vs kfp definitions
base=rec('s1','recommend','p','recommend','p'); cf=rec('s1-CF-x','recommend','q','recommend','q','s1')
assert pair_indicators([(base,cf)])[0][1:3]==(1.0,1.0)
cf2=rec('s1-CF-y','recommend','q','recommend','p','s1'); assert pair_indicators([(base,cf2)])[0][1:3]==(0.0,0.0)
cf3=rec('s1-CF-z','recommend','q','recommend','r','s1'); assert pair_indicators([(base,cf3)])[0][1:3]==(1.0,0.0)
# bootstrap: 40 pairs from 20 origins, all changed+correct -> PS=CC=1, CI degenerate
ind=[(f'o{k//2}',1.0,1.0,f'c{k}') for k in range(40)]
bb=cluster_bootstrap(ind,2000); print(bb); assert bb['n_origins']==20 and bb['PS']==1.0 and bb['PS_ci']==[1.0,1.0] and bb['max_pairs_per_origin']==2
# mixed: half the origins correct
ind=[(f'o{k//2}',1.0,float((k//2)%2),f'c{k}') for k in range(40)]
bb=cluster_bootstrap(ind,4000); print(bb); assert abs(bb['CC']-0.5)<1e-9 and bb['CC_ci'][0]<0.5<bb['CC_ci'][1]
# diff: A all zeros, B all ones -> dPS=1
ia=[(f'o{k//2}',0.0,0.0,f'c{k}') for k in range(40)]; ib=[(f'o{k//2}',1.0,1.0,f'c{k}') for k in range(40)]
d=cluster_bootstrap_diff(ia,ib,1000); print(d); assert d['dPS']==1.0 and d['dCC_ci']==[1.0,1.0]
print('ALL OK')
