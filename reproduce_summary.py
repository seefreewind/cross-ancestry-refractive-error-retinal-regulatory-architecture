from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from scipy.stats.contingency import odds_ratio
root=Path(__file__).resolve().parent
table=[[71,1161],[19653,359626]]
value,p=fisher_exact(table)
ci=odds_ratio(table,kind='conditional').confidence_interval(.95)
assert round(value,3)==1.119 and round(p,3)==.367
assert (round(ci.low,3),round(ci.high,3))==(.867,1.423)
t=pd.read_csv(root/'results/phase2/PHASE2_DAR_HETEROGENEITY_ENRICHMENT.tsv',sep='\t')
row=t.loc[t.endpoint.eq('matched_permutation_mean_chi2_het_difference')].iloc[0]
n=pd.read_csv(root/'results/phase2/PHASE2_DAR_MATCHED_PERMUTATION_NULL.tsv',sep='\t')
perm=(1+int((n.null_mean_difference.abs()>=abs(row.observed_mean_difference)).sum()))/(len(n)+1)
assert round(perm,3)==.334
b=pd.read_csv(root/'results/iovs_final_qc/BLOCK_BOOTSTRAP_INPUT.tsv',sep='\t')
v=b[['dar_N','comparator_N','dar_events','comparator_events']].to_numpy()
n1,n0,a,c=v[np.random.default_rng(20260912).integers(0,200,size=(2000,200))].sum(axis=1).T
lo,hi=np.quantile(a*(n0-c)/((n1-a)*c),[.025,.975])
assert (round(lo,3),round(hi,3))==(.821,1.438)
print(f'PASS: OR={value:.3f}; exact CI={ci.low:.3f}-{ci.high:.3f}; Fisher P={p:.3f}; matched P={perm:.3f}; bootstrap CI={lo:.3f}-{hi:.3f}')
