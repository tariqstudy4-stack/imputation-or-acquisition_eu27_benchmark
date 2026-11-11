# Aggregates results/center_<pattern>_0_500.csv into results/center_summary.json (run from the repository root).
import json,pandas as pd
out={}
for pat in ('MCAR','FLOW','BLOCK'):
    f=pd.read_csv(f'results/center_{pat}_0_500.csv')
    for (rate,c),d in f.groupby(['rate','centre']):
        out[f'{pat}|{rate}|{c}']=dict(mdape_med=100*d.mdape.median(),cov_mean=100*d.coverage.mean(),width_med=100*d.width.median(),flips=100*d.flips.mean(),n=len(d))
json.dump(out,open('results/center_summary.json','w'),indent=1); print(len(out))
