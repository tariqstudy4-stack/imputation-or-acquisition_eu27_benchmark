# Additional summaries of the 500-seed main grid: mean MdAPE, median coverage, and a 95% bootstrap
# interval over seeds for the ranking-flip rate. Writes ../results/summary_extra.json.
import glob, json, os
import numpy as np, pandas as pd
R = os.path.join('..', 'results')
r = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(R, 'res_*_0_500.csv'))])
rng = np.random.default_rng(0)
def boot(x, B=2000):
    x = np.asarray(x, float)
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)])
    return np.percentile(m, [2.5, 97.5])
out = {}
for (m, p, rate), g in r.groupby(['method', 'pattern', 'rate']):
    lo, hi = boot(g.flips.values)
    cov = g.coverage.dropna()
    out['%s|%s|%s' % (m, p, rate)] = dict(
        mdape_med=100 * g.mdape.median(), mdape_mean=100 * g.mdape.mean(),
        cov_mean=None if cov.empty else 100 * cov.mean(), cov_med=None if cov.empty else 100 * cov.median(),
        flips=100 * g.flips.mean(), flips_lo=100 * lo, flips_hi=100 * hi, n=int(len(g)))
json.dump(out, open(os.path.join(R, 'summary_extra.json'), 'w'), indent=1)
print('wrote', len(out), 'conditions')
