from __future__ import annotations
import base64, csv, hashlib, json, re, statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
TABLES = OUT / 'tables' / 'five_seed'
FIGURES = OUT / 'figures' / 'five_seed'
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
FILES = {
 'FS01_CU':'itransformer-fs01-cu-kaggle-final.ipynb',
 'FS02_PAPER_CU':'itransformer-fs02-paper-cu-kaggle-final.ipynb',
 'FS03_PAPER_EXO':'itransformer-fs03-paper-exo-kaggle-final.ipynb',
 'FS04_METALS_CU':'itransformer-fs04-metals-cu-kaggle-final.ipynb',
 'FS05_METALS_EXO':'itransformer-fs05-metals-exo-kaggle-final.ipynb',
 'FS06_WTI_CU':'itransformer-fs06-wti-cu-kaggle-final.ipynb',
}
DONE = re.compile(r'DONE\s+(FS\d+_[A-Z0-9_]+)\s+(0\.\d+)\s+(\w+)\s+(itransformer_direct|itransformer_residual|naive)\s+seed=(\d+|shared)\s+MAE=([0-9.]+)')

def out_text(cell):
    return ''.join(''.join(o.get('text', [])) for o in cell.get('outputs', []) if o.get('text') is not None)

def save_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w=csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); w.writeheader(); w.writerows(rows)

all_rows=[]; statuses=[]; figures=[]; hashes={}
for expected_fs, filename in FILES.items():
    p=ROOT/'notebooks'/filename; raw=p.read_bytes(); hashes[filename]=hashlib.sha256(raw).hexdigest()
    nb=json.loads(raw.decode('utf-8-sig')); cells=nb.get('cells', [])
    text='\n'.join(out_text(c) for c in cells)
    found={}
    for m in DONE.finditer(text):
        fs,ratio,scheme,model,seed,mae=m.groups()
        key=(fs,ratio,scheme,model,seed)
        found[key]={'feature_set':fs,'ratio':f'{float(ratio):.3f}','scheme':scheme,'model':model,
                    'objective':model.rsplit('_',1)[-1] if model.startswith('itransformer_') else 'baseline',
                    'seed':seed,'mae':f'{float(mae):.5f}','metric_source':'saved notebook cell output log','notebook':filename}
    rows=list(found.values()); all_rows.extend(rows)
    sha=re.search(r'SHA256:\s*([0-9a-f]{64})', text)
    cell18=out_text(cells[18]) if len(cells)>18 else ''
    final=re.search(r'Run state:\s*(\w+)\s*\|\s*neural:\s*(\d+)\s*/\s*(\d+)\s*\|\s*Naive:\s*(\d+)\s*/\s*(\d+)', cell18)
    saved=re.findall(r'"state"\s*:\s*"([^"]+)"[\s\S]*?"expected_neural_fits"\s*:\s*(\d+)[\s\S]*?"completed_neural_fits"\s*:\s*(\d+)[\s\S]*?"expected_naive_baselines"\s*:\s*(\d+)[\s\S]*?"completed_naive_baselines"\s*:\s*(\d+)', text)
    if final:
        state,dn,en,db,eb=final.groups(); status_source='final notebook status output'
    elif saved:
        state,en,dn,eb,db=saved[-1]; status_source='last saved run_status JSON in notebook output'
    else:
        state,dn,en,db,eb='unknown','','120','','12'; status_source='status not found in saved output'
    counts={model:sum(r['model']==model for r in rows) for model in ['itransformer_direct','itransformer_residual','naive']}
    statuses.append({'feature_set':expected_fs,'notebook':filename,'run_state':state,
        'completed_neural_fits_checkpoint':dn,'expected_neural_fits':en,
        'completed_naive_checkpoint':db,'expected_naive_baselines':eb,
        'direct_mae_rows_in_saved_log':counts['itransformer_direct'],
        'residual_mae_rows_in_saved_log':counts['itransformer_residual'],
        'naive_mae_rows_in_saved_log':counts['naive'],'data_sha256':sha.group(1) if sha else '',
        'status_source':status_source})
    if len(cells)>16:
        for o in cells[16].get('outputs',[]):
            png=o.get('data',{}).get('image/png')
            if png:
                if isinstance(png,list): png=''.join(png)
                name=f'{expected_fs}_5seed_r0900_expanding.png'
                (FIGURES/name).write_bytes(base64.b64decode(png))
                figures.append({'feature_set':expected_fs,'figure':f'outputs/figures/five_seed/{name}',
                    'source_notebook':filename,'description':'Saved notebook plot: actual Copper and 5-seed mean prediction; 90/10 expanding'})
                break

all_rows.sort(key=lambda r:(r['feature_set'],float(r['ratio']),r['scheme'],r['model'],r['seed']))
save_csv(TABLES/'job_mae_seed41_45.csv',all_rows,['feature_set','ratio','scheme','model','objective','seed','mae','metric_source','notebook'])
save_csv(TABLES/'run_status.csv',statuses,['feature_set','notebook','run_state','completed_neural_fits_checkpoint','expected_neural_fits','completed_naive_checkpoint','expected_naive_baselines','direct_mae_rows_in_saved_log','residual_mae_rows_in_saved_log','naive_mae_rows_in_saved_log','data_sha256','status_source'])
save_csv(TABLES/'figure_manifest.csv',figures,['feature_set','figure','source_notebook','description'])
complete={s['feature_set'] for s in statuses if s['run_state']=='complete' and s['completed_neural_fits_checkpoint']==s['expected_neural_fits']=='120' and s['completed_naive_checkpoint']==s['expected_naive_baselines']=='12'}
condition=defaultdict(list)
for r in all_rows:
    if r['feature_set'] in complete: condition[(r['feature_set'],r['ratio'],r['scheme'],r['model'])].append(float(r['mae']))
condition_rows=[]
for (fs,ratio,scheme,model),vals in sorted(condition.items()):
    condition_rows.append({'feature_set':fs,'ratio':ratio,'scheme':scheme,'model':model,'n_logged_jobs':len(vals),'mean_mae':f'{statistics.mean(vals):.6f}','std_mae':f'{statistics.stdev(vals):.6f}' if len(vals)>1 else ''})
save_csv(TABLES/'condition_summary_fs01_fs05.csv',condition_rows,['feature_set','ratio','scheme','model','n_logged_jobs','mean_mae','std_mae'])
summary=[]
for fs in sorted(complete):
    rr=[r for r in all_rows if r['feature_set']==fs]
    direct=[float(r['mae']) for r in rr if r['model']=='itransformer_direct']
    residual=[float(r['mae']) for r in rr if r['model']=='itransformer_residual']
    naive={(r['ratio'],r['scheme']):float(r['mae']) for r in rr if r['model']=='naive'}
    resid_rows=[r for r in rr if r['model']=='itransformer_residual']
    wins=sum(float(r['mae'])<naive[(r['ratio'],r['scheme'])] for r in resid_rows)
    skills=[100*(naive[(r['ratio'],r['scheme'])]-float(r['mae']))/naive[(r['ratio'],r['scheme'])] for r in resid_rows]
    groups=defaultdict(list)
    for r in rr:
        if r['model']!='naive': groups[(r['ratio'],r['scheme'],r['model'])].append(float(r['mae']))
    sd_d=[statistics.stdev(v) for (a,b,m),v in groups.items() if m=='itransformer_direct' and len(v)>1]
    sd_r=[statistics.stdev(v) for (a,b,m),v in groups.items() if m=='itransformer_residual' and len(v)>1]
    summary.append({'feature_set':fs,'status':'complete','direct_runs':len(direct),'mean_direct_mae':f'{statistics.mean(direct):.6f}','mean_within_config_seed_std_direct':f'{statistics.mean(sd_d):.6f}',
      'residual_runs':len(residual),'mean_residual_mae':f'{statistics.mean(residual):.6f}','mean_within_config_seed_std_residual':f'{statistics.mean(sd_r):.6f}',
      'naive_configs':len(naive),'mean_naive_mae':f'{statistics.mean(naive.values()):.6f}','residual_wins_vs_naive':f'{wins}/{len(residual)}',
      'mean_residual_skill_vs_naive_pct':f'{statistics.mean(skills):.4f}','data_sha256':next(s['data_sha256'] for s in statuses if s['feature_set']==fs)})
save_csv(TABLES/'feature_set_summary_fs01_fs05.csv',summary,['feature_set','status','direct_runs','mean_direct_mae','mean_within_config_seed_std_direct','residual_runs','mean_residual_mae','mean_within_config_seed_std_residual','naive_configs','mean_naive_mae','residual_wins_vs_naive','mean_residual_skill_vs_naive_pct','data_sha256'])
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if summary:
        labels=[r['feature_set'].replace('_','\n',1) for r in summary]; x=list(range(len(labels))); w=.36
        nd=[float(r['mean_direct_mae']) for r in summary]; nr=[float(r['mean_naive_mae']) for r in summary]; rs=[float(r['mean_residual_mae']) for r in summary]
        fig,axes=plt.subplots(1,2,figsize=(13,5.5),constrained_layout=True)
        axes[0].bar([i-w/2 for i in x],nd,w,label='iTransformer Direct',color='#d95f02'); axes[0].bar([i+w/2 for i in x],nr,w,label='Naive',color='#7570b3')
        axes[0].set_title('Direct vs Naive'); axes[0].set_ylabel('MAE (giá Copper, thang đo gốc)'); axes[0].legend(frameon=False)
        axes[1].bar([i-w/2 for i in x],rs,w,label='iTransformer Residual',color='#1b9e77'); axes[1].bar([i+w/2 for i in x],nr,w,label='Naive',color='#7570b3')
        axes[1].set_title('Residual vs Naive'); axes[1].set_ylabel('MAE (giá Copper, thang đo gốc)'); axes[1].legend(frameon=False)
        for ax in axes: ax.set_xticks(x,labels); ax.grid(axis='y',alpha=.25); ax.set_axisbelow(True)
        fig.suptitle('MAE trung bình trên 5 feature set hoàn tất (seed 41–45; 60 lần chạy/model)')
        fig.savefig(FIGURES/'mean_mae_by_feature_set_fs01_fs05.png',dpi=180); plt.close(fig)
except ImportError:
    pass
manifest={'purpose':'Extract saved MAE logs and notebook figures only; no model training or new prediction was run.',
  'complete_feature_sets':sorted(complete),'feature_sets':statuses,'job_mae_rows':len(all_rows),'notebook_sha256':hashes,
  'outputs':['outputs/tables/five_seed/job_mae_seed41_45.csv','outputs/tables/five_seed/run_status.csv','outputs/tables/five_seed/condition_summary_fs01_fs05.csv','outputs/tables/five_seed/feature_set_summary_fs01_fs05.csv','outputs/tables/five_seed/figure_manifest.csv','outputs/figures/five_seed/']}
(TABLES/'extraction_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(f'Exported {len(all_rows)} saved MAE log rows; complete feature sets: {", ".join(sorted(complete))}')
for s in statuses: print(s['feature_set'],s['run_state'],f"checkpoint {s['completed_neural_fits_checkpoint']}/{s['expected_neural_fits']} neural, {s['completed_naive_checkpoint']}/{s['expected_naive_baselines']} baseline; saved-log rows {s['direct_mae_rows_in_saved_log']}/{s['residual_mae_rows_in_saved_log']}/{s['naive_mae_rows_in_saved_log']}")
