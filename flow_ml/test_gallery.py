"""Build a local browser gallery and Markdown index for every test prediction."""
import argparse
import json
from pathlib import Path
import numpy as np

def build(folder):
    folder=Path(folder)
    report=json.loads((folder/'final_metrics.json').read_text())
    old={r['seed']:r for r in report['test']['old_gnn']['cases']}
    rows=sorted(report['test']['multiscale']['cases'],key=lambda r:r['seed'])
    data=[]
    lines=['# All 40 test cases','',
           'Each row links separate solver, previous GNN, multiscale GNN, and signed-error pressure/quiver plots. Solver and prediction scales match within each case. Error plots use a separate symmetric pressure scale.',
           '', '[Open the interactive gallery](test_gallery.html)','',
           '| Seed | Solver | Previous GNN | Multiscale GNN | Error |','|---|---|---|---|---|']
    for row in rows:
        seed=row['seed']
        images={kind:f'test_plots/seed_{seed}_{kind}.png' for kind in ['solver','old_gnn','multiscale','error']}
        for path in images.values():
            if not (folder/path).is_file():
                raise ValueError(f'Missing test plot: {path}')
        data.append({'seed':seed,'images':images,'old':old[seed]['loss'],'new':row['loss'],
                     'old_p_rmse':float(np.sqrt(old[seed]['mse'][2])*.0025),
                     'new_p_rmse':float(np.sqrt(row['mse'][2])*.0025)})
        lines.append(f'| {seed} | [Solver]({images["solver"]}) | [Previous]({images["old_gnn"]}) | [New]({images["multiscale"]}) | [Error]({images["error"]}) |')
    (folder/'test_index.md').write_text('\n'.join(lines)+'\n')
    html='''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Flow GNN — 40 unseen test cases</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:#172033;font:15px system-ui,sans-serif}
header{padding:22px 30px;background:white;border-bottom:1px solid #dbe2ea}h1{font-size:24px;margin:0 0 8px}
p{margin:5px 0;color:#526174;line-height:1.5}.controls{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-top:18px}
button,select{font:inherit;background:white;color:#172033;padding:9px 13px;border:1px solid #b8c5d6;border-radius:7px;cursor:pointer}
button.active{background:#1d4ed8;color:white;border-color:#1d4ed8}.metrics{margin:16px 30px;display:flex;gap:24px;flex-wrap:wrap;color:#34445b}
main{display:grid;grid-template-columns:1fr 1fr;gap:16px;padding:0 22px 24px}figure{margin:0;background:white;border:1px solid #dbe2ea;border-radius:10px;overflow:hidden}
figcaption{font-weight:650;padding:14px 18px;border-bottom:1px solid #eef1f6}img{display:block;width:100%;height:auto}a{color:#1d4ed8}
footer{padding:0 30px 24px;color:#526174}@media(max-width:800px){main{grid-template-columns:1fr}header{padding:20px}.metrics{margin:16px 20px}}
</style><header><h1>40 unseen flow cases</h1>
<p>300 training cases · 40 validation cases · 40 test cases. Checkpoint selected using validation only.</p>
<p>Click a figure to open its full-size PNG. Solver and prediction colors and arrows share scales within each case.</p>
<div class="controls"><button id="prev">← Previous case</button><label>Seed <select id="case"></select></label><button id="next">Next case →</button>
<button data-kind="multiscale" class="active">New GNN</button><button data-kind="old_gnn">Previous GNN</button><button data-kind="error">Signed error</button></div></header>
<div class="metrics" id="metrics"></div><main><figure><figcaption>Solver target</figcaption><a id="solverLink" target="_blank"><img id="solver" alt="Solver pressure and velocity"></a></figure>
<figure><figcaption id="predictionTitle">New GNN</figcaption><a id="predictionLink" target="_blank"><img id="prediction" alt="Predicted pressure and velocity"></a></figure></main>
<footer>Signed-error views use their own symmetric pressure scale; arrows show velocity error. All fields exclude solid cells. <a href="test_index.md">All image links</a> · <a href="summary.md">Experiment report</a></footer>
<script>const cases=__DATA__;let index=0,kind='multiscale';
const selector=document.getElementById('case');
for(let i=0;i<cases.length;i++){const o=document.createElement('option');o.value=i;o.textContent=cases[i].seed;selector.appendChild(o)}
function draw(){const c=cases[index];selector.value=index;
document.getElementById('solver').src=c.images.solver;document.getElementById('solverLink').href=c.images.solver;
document.getElementById('prediction').src=c.images[kind];document.getElementById('predictionLink').href=c.images[kind];
document.getElementById('predictionTitle').textContent=({multiscale:'New multiscale GNN',old_gnn:'Previous 140-epoch GNN',error:'New GNN minus solver'})[kind];
document.getElementById('metrics').textContent=`Case ${index+1}/${cases.length} · Field MSE: old ${c.old.toExponential(3)} → new ${c.new.toExponential(3)} · Pressure RMSE: old ${c.old_p_rmse.toExponential(3)} → new ${c.new_p_rmse.toExponential(3)} (lattice)`;
document.querySelectorAll('[data-kind]').forEach(b=>b.classList.toggle('active',b.dataset.kind===kind));}
selector.onchange=()=>{index=+selector.value;draw()};document.getElementById('prev').onclick=()=>{index=(index+cases.length-1)%cases.length;draw()};
document.getElementById('next').onclick=()=>{index=(index+1)%cases.length;draw()};document.querySelectorAll('[data-kind]').forEach(b=>b.onclick=()=>{kind=b.dataset.kind;draw()});
document.addEventListener('keydown',e=>{if(e.target.tagName==='SELECT')return;if(e.key==='ArrowRight')document.getElementById('next').click();if(e.key==='ArrowLeft')document.getElementById('prev').click()});draw();
</script></html>'''.replace('__DATA__',json.dumps(data))
    (folder/'test_gallery.html').write_text(html)
    print(f'Gallery and index verified: {len(rows)} cases, {len(rows)*4} PNGs.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path)
    build(p.parse_args().folder)
