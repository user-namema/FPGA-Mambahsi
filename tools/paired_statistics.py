"""Reanalyse existing paired seed results. No new training or inference.
95% Student-t CI (df=9); exact two-sided paired sign-flip test (2^10).
Holm adjustment across the eight primary OA comparisons (2 contrasts x 4 datasets).
"""
from pathlib import Path
import csv,json,itertools,math,statistics
R=Path(__file__).resolve().parents[1]
a=list(csv.DictReader((R/'evidence/paired_accuracy_by_seed.csv').open(encoding='utf-8-sig')))
q=json.loads((R/'evidence/qat_metrics.json').read_text())['rows']
DS=['UP','HanChuan','HongHu','Houston'];out=[]
def summary(ds,contrast,values):
 assert len(values)==10 and all(math.isfinite(v) for v in values)
 m=statistics.mean(values);sd=statistics.stdev(values);h=2.2621571628540993*sd/math.sqrt(10)
 p=sum(abs(sum(v*s for v,s in zip(values,sgn))/10)>=abs(m)-1e-12 for sgn in itertools.product([-1,1],repeat=10))/1024
 return dict(dataset=ds,contrast=contrast,n=10,mean_pp=m,sd_pp=sd,ci95_low=m-h,ci95_high=m+h,p_signflip=p,values=values)
for ds in DS:
 aa=sorted([r for r in a if r['dataset']==ds],key=lambda r:int(r['seed']))
 qq=sorted([r for r in q if r['dataset']==ds],key=lambda r:int(r['seed']))
 assert [int(r['seed']) for r in aa]==list(range(10))
 assert [int(r['seed']) for r in qq]==list(range(10))
 out.append(summary(ds,'A_cost_per_channel_minus_shared',[float(r['OA_cost_shared_pp']) for r in aa]))
 out.append(summary(ds,'QAT_loss_FP32_minus_QAT8',[r['fp32_OA']-r['qat_saved_batch_OA'] for r in qq]))
prior=0
for j,i in enumerate(sorted(range(8),key=lambda i:out[i]['p_signflip'])):
 prior=max(prior,min(1,(8-j)*out[i]['p_signflip']));out[i]['p_holm_8']=prior
(R/'evidence/paired_inference.json').write_text(json.dumps(out,indent=2))
with (R/'evidence/paired_inference.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=[k for k in out[0] if k!='values']);w.writeheader();w.writerows({k:v for k,v in r.items() if k!='values'} for r in out)
for r in out:print(r['dataset'],r['contrast'],f"{r['mean_pp']:.3f} [{r['ci95_low']:.3f},{r['ci95_high']:.3f}], pH={r['p_holm_8']:.3f}")
