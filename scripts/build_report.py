"""Build a self-contained HTML report and exportable scientific figures."""
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parents[1]
REPOSITORY_URL='https://github.com/RuiyuM/pbae-cellular-uptake-reproduction'
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import base64
import json
import html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS=['LM','RF','GBT','NN']
LABELS={'LM':'Linear regression','RF':'Random forest','GBT':'Gradient boosting','NN':'Neural network'}
summary=pd.read_csv(ROOT/'results/table2_comparison.csv')
protocol=json.loads((ROOT/'results/protocol.json').read_text())
verification=json.loads((ROOT/'results/verification.json').read_text())
reconstruction=json.loads((ROOT/'data/processed/reconstruction_metadata.json').read_text())
figdir=ROOT/'figures';figdir.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                     'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})

def select(experiment):
    return summary[summary.experiment==experiment].set_index('model').loc[MODELS]

primary=select('paper17_author_split'); varied=select('paper17_varied_splits'); exact=select('paper17_exact_symbol')
fig,ax=plt.subplots(figsize=(10.5,5.8))
x=np.arange(4)
ax.scatter(x-.2,primary.paper_mae,marker='D',s=65,color='#182a3a',label='Paper Table 2 (n=5)',zorder=5)
ax.errorbar(x,primary['mean'],yerr=primary['std'],fmt='o',color='#087f8c',capsize=5,
            markersize=8,label='Reconstructed 17 inputs: fixed split (n=5)')
ax.errorbar(x+.2,varied['mean'],yerr=varied['std'],fmt='s',color='#bd6b17',capsize=5,
            markersize=7,label='Reconstructed 17 inputs: varied splits (n=5)')
ax.set_xticks(x,[LABELS[m] for m in MODELS]);ax.set_ylabel('Test MAE (uptake percentage points)')
ax.set_ylim(0,22);ax.set_xlim(-.5,3.5);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
ax.legend(frameon=False,loc='upper right',fontsize=9)
ax.set_title('pBAE cellular uptake: rerun versus published Table 2',loc='left',pad=18,fontweight='bold')
fig.text(.125,.02,'Error bars: sample SD, not confidence intervals. Paper Table 2 supplies means only.\nReconstructed inputs and documented assumptions; not an exact recovery of the published experiment.',fontsize=9,color='#475569')
fig.subplots_adjust(bottom=.2)
for ext in ['png','svg']:fig.savefig(figdir/f'table2_comparison.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)

history=pd.read_csv(ROOT/'results/nn_training_history.csv')
hh=history[history.run_id.str.startswith('paper17_author_split_')]
fig,ax=plt.subplots(figsize=(10.5,4.6))
for column,label,color in [('mae','Training MAE','#087f8c'),('val_mae','Validation MAE','#bd6b17')]:
    stat=hh.groupby('epoch')[column].agg(['mean','std']);xx=stat.index.to_numpy()
    ax.plot(xx,stat['mean'],label=label,color=color,linewidth=2)
    ax.fill_between(xx,(stat['mean']-stat['std']).to_numpy(),(stat['mean']+stat['std']).to_numpy(),color=color,alpha=.14)
ax.set(xlabel='Epoch',ylabel='MAE (uptake percentage points)',xlim=(1,100),ylim=(0,None))
ax.set_title('Neural network: 17 inputs, 41 + 41 hidden units',loc='left',fontweight='bold')
ax.legend(frameon=False);ax.grid(alpha=.16)
fig.text(.125,.005,'Mean and sample SD over five initializations; fixed split=7. Final epoch evaluated, no test-based epoch selection.',fontsize=9,color='#475569')
fig.subplots_adjust(bottom=.2)
for ext in ['png','svg']:fig.savefig(figdir/f'nn_learning_curves.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)

genes=['CLTA','CLTB','CLTC','PICALM','CAV1','CAV2'];cells=['OVCAR4','PANC02','4T1','HDF']
g=pd.read_csv(ROOT/'data/processed/recovered_six_genes_by_cell.csv')
gg=g[g.matching=='repository_matching'].set_index('CellType').loc[cells,genes].T
fig,ax=plt.subplots(figsize=(8.2,5.8));im=ax.imshow(gg.to_numpy(),cmap='RdBu_r',vmin=-4,vmax=4,aspect='auto')
ax.set_xticks(np.arange(4),cells);ax.set_yticks(np.arange(6),genes)
for i in range(6):
    for j in range(4):
        v=gg.iloc[i,j];ax.text(j,i,f'{v:.2f}',ha='center',va='center',color='white' if abs(v)>2.5 else '#172333',fontsize=12)
ax.set_title('Six genes recovered from the four cited GEO arrays',loc='left',fontweight='bold',pad=16)
fig.colorbar(im,ax=ax,label='log2(normalized mean signal)',shrink=.85)
fig.text(.125,.02,'Repository-compatible matching; CLTC also matches CLTCL1 in human arrays.\nThe mouse CAV1/CAV2 directions differ from the published Figure 3B.',fontsize=9,color='#475569')
fig.subplots_adjust(bottom=.16)
for ext in ['png','svg']:fig.savefig(figdir/f'recovered_gene_expression.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)

def value(row):return f"{row['mean']:.2f} ± {row['std']:.2f}"
comparison=pd.DataFrame({
    '模型':['线性回归 LM','随机森林 RF','梯度提升树 GBT','神经网络 NN'],
    '论文 Table 2':[f'{v:.2f}' for v in primary.paper_mae],
    '固定划分：均值 ± SD':[value(primary.loc[m]) for m in MODELS],
    '更换划分：均值 ± SD':[value(varied.loc[m]) for m in MODELS],
    '固定划分相对论文差值':[f'{primary.loc[m,"difference_vs_paper"]:+.2f}' for m in MODELS],
})
comparison.to_csv(ROOT/'results/table2_comparison_readable.csv',index=False)
probe_table=gg.T.reset_index().rename(columns={'CellType':'细胞'})
for c in genes:probe_table[c]=probe_table[c].map(lambda v:f'{v:.6f}')
exact_table=pd.DataFrame({'模型':MODELS,'作者兼容匹配':[value(primary.loc[m]) for m in MODELS],
                        '精确基因名称匹配':[value(exact.loc[m]) for m in MODELS]})
diag=pd.read_csv(ROOT/'results/linear_numerical_diagnostics.csv')
stable=diag[diag.experiment=='paper17_author_split'].stable_svd_test_mae.mean()
rf_alt=summary[summary.experiment=='paper17_rf_all_features'].iloc[0]
sources=''.join(f'<li>{cell}：<a href="{data["url"]}">{data["accession"]}</a></li>' for cell,data in reconstruction['samples'].items())
def inline_image(name):
    payload=base64.b64encode((figdir/name).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{payload}" alt="{name}" />'
body=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>pBAE 论文复跑与 Table 2 对比</title><style>
body{{margin:0;background:#eef3f6;color:#182a3a;font:16px/1.8 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}}
main{{max-width:1080px;margin:40px auto;padding:46px 54px;background:white;border-radius:18px;box-shadow:0 12px 45px #182a3a0b}}
h1{{font-size:32px;line-height:1.35;margin:10px 0 14px}}h2{{font-size:23px;margin:38px 0 12px;border-top:1px solid #dbe5eb;padding-top:26px}}
h3{{font-size:18px;margin:24px 0 6px}}p{{margin:12px 0}}.eyebrow{{color:#087f8c;font-size:13px;font-weight:700;letter-spacing:2px}}
.lead{{font-size:19px}}.note{{background:#fff7e8;border-left:4px solid #bd6b17;padding:14px 20px;border-radius:4px}}
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:26px 0}}.card{{background:#f0f7f8;padding:18px;border-radius:10px}}
.card b{{display:block;font-size:27px;color:#087f8c}}.small,footer{{font-size:13px;color:#526779}}a{{color:#087f8c}}
table{{border-collapse:collapse;width:100%;font-size:14px;margin:16px 0;white-space:nowrap}}th{{text-align:left;background:#eaf2f5;color:#193b4d}}td,th{{padding:11px 12px;border-bottom:1px solid #dce7ec}}tr:nth-child(even){{background:#f8fafb}}
.scroll{{overflow-x:auto}}img{{width:100%;height:auto;margin:18px 0}}code,pre{{background:#f1f5f7;border-radius:5px;font-size:13px}}code{{padding:2px 5px}}pre{{padding:16px;overflow:auto;line-height:1.6}}
li{{margin:6px 0}}footer{{border-top:1px solid #dbe5eb;padding-top:20px;margin-top:30px}}@media(max-width:760px){{main{{margin:0;padding:24px;border-radius:0}}.cards{{grid-template-columns:repeat(2,1fr)}}h1{{font-size:27px}}}}
</style></head><body><main>
<div class="eyebrow">REPRODUCTION STUDY · 2026-10-20</div><h1>pBAE 细胞摄取预测<br>补齐基因、重跑四模型、对比 Table 2</h1>
<p class="lead">已完成缺失基因恢复和四模型训练。恢复了论文所列的 17 个特征，并按公开最终参数运行；这属于有来源与校验记录的近似复现，仍不能宣称精确重现论文实验。</p>
<p class="repository"><strong>复现代码、数据与结果：</strong><br><a href="{REPOSITORY_URL}">{REPOSITORY_URL}</a></p>
<div class="cards"><div class="card"><b>240</b>原始摄取样本</div><div class="card"><b>17</b>重建模型输入</div><div class="card"><b>212 / 212</b>表达值校验通过</div><div class="card"><b>16</b>完整 NN 训练</div></div>
<h2>与论文 Table 2 的对比</h2><p>指标为测试集平均绝对误差 MAE，单位是摄取率的百分点，越低越好。我们的 ± 为 5 次运行的样本标准差，不能直接当作置信区间。论文 Table 2 仅提供 5 次平均值。</p>
<div class="scroll">{comparison.to_html(index=False,escape=True,border=0)}</div>
<p><strong>固定划分</strong>沿用仓库的划分种子 7，只改变模型随机种子 7–11；<strong>更换划分</strong>使用划分种子 7–11，并使用对应模型种子。每次均为 192 条训练、24 条验证、24 条测试。所有结果均保留，没有筛选最接近论文的一次。</p>
<p>固定划分下，RF 与论文均值差 {primary.loc['RF','difference_vs_paper']:+.2f}，NN 差 {primary.loc['NN','difference_vs_paper']:+.2f}，GBT 差 {primary.loc['GBT','difference_vs_paper']:+.2f}。更换划分后的 GBT 和 NN 均值接近，但波动明显；这不能证明与论文具有统计等价性。</p>
{inline_image('table2_comparison.png')}
<h2>缺失基因找到了什么</h2><p>从论文列出的四个 GEO 原始芯片文件中，恢复了缺失的 CLTA、CLTB、CLTC、PICALM，并重新提取 CAV1、CAV2。没有使用零值、随机值或从热图估读的数值填补。</p>
<ul>{sources}</ul>
<p>通过已有 53 个基因 × 4 类细胞反向校验，以下处理规则在 212 个数值上全部匹配当前作者仓库，最大绝对误差 <code>{reconstruction['validation']['max_absolute_error']:.2e}</code>：</p>
<pre>expression = log2(median(所匹配探针的 gMeanSignal)
                  / mean(整张芯片所有点的 gMeanSignal))</pre>
<p>匹配规则为不区分大小写的基因名称子串匹配；HDF 的 AGRIN 使用别名 AGRN。该规则由可验证数据推断得到，作者未公开对应预处理脚本。</p>
<div class="scroll">{probe_table.to_html(index=False,escape=True,border=0)}</div>
{inline_image('recovered_gene_expression.png')}
<div class="note"><strong>重要限制：</strong>该恢复过程匹配的是当前仓库的表达数据。恢复的两种小鼠细胞 CAV1/CAV2 方向与论文 Figure 3B 的颜色不一致；论文最终的六基因输入矩阵和具体探针选择仍未找到。我们保留原始来源计算值，没有人为调整数据以贴近论文。</div>
<h2>本次采用的训练设置</h2>
<ul><li>输入：6 个聚合物比例、Size、Polydispersity、6 个基因，以及癌变、转移、人类物种三个二元特征。细胞身份只用于合并和记录，不额外作为特征。</li>
<li>RF：75 棵树、最大深度 13；未公开的 <code>max_features</code> 沿用仓库的 <code>sqrt</code>。</li>
<li>GBT：50 棵树、最大深度 5、学习率 0.1。</li>
<li>NN：17 → 41 ReLU → 41 ReLU → 1 线性输出；RMSprop 学习率 0.001，MSE 损失，batch size 5，100 epochs。优化器和批大小取自作者最终训练代码。</li>
<li>所有标准化统计量仅从训练集计算。使用最后一轮 NN 模型，不根据测试结果选轮次，不裁剪预测值。直接使用论文已报告参数，没有重做历史调参搜索。</li>
<li>环境：Python {protocol['environment']['python']}，scikit-learn 1.2.2，TensorFlow/Keras 2.10.0，NumPy 1.23.5，SciPy 1.10.1；CPU 训练。</li></ul>
{inline_image('nn_learning_curves.png')}
<h2>未公开细节与敏感性检查</h2>
<p>论文正文称 NN 输入层为 9 个节点，但 Figure 5 明确列出 17 个特征；本次采用 17 个特征。论文没有给出 5 次实验的种子和重复方式，因此同时报告固定划分和更换划分。期刊旧补充材料地址在本次下载时返回 404，备用地址返回 403。</p>
<h3>基因匹配</h3><p>与当前仓库一致的子串匹配会把人类的 CLTCL1 也匹配到 CLTC。我们另外运行了精确名称匹配，以单独显示这一影响；没有根据测试效果选择主结果。</p>
<div class="scroll">{exact_table.to_html(index=False,escape=True,border=0)}</div>
<h3>随机森林未公开参数</h3><p>将 <code>max_features</code> 改为 1.0 的额外对照，固定划分 MAE 为 {value(rf_alt)}。该参数足以改变与论文的数值差距，因此不能仅靠调到接近 13.09 就宣称复现成功。</p>
<h3>线性回归的数值稳定性</h3><p>这些特征存在共线性：配方比例总和固定，细胞相关表达只有四组独特取值。原版 scikit-learn 的求解产生很大的系数。保持报告的 LM 结果不变，同时提供了单独的稳定 SVD 诊断；固定划分的稳定解 MAE 为 {stable:.2f}。这也是旧库版本及特征排列可能影响复跑数值的原因。</p>
<h2>验证与可重跑文件</h2><p>已检查全部 1,536 条测试预测与报告 MAE 一致，16 组划分互不泄漏，16 次 NN 均训练满 100 轮。相同数据与种子的独立运行预测完全一致；保存的 NN 重新载入后能重现预测。</p>
<ul><li><a href="results/table2_comparison.csv">全部对比结果 CSV</a> / <a href="results/all_runs.csv">逐次实验结果</a> / <a href="results/test_predictions.csv">测试预测</a></li>
<li><a href="data/processed/paper17_repository_matching.csv">重建的 240 × 17 数据</a> / <a href="data/processed/recovered_six_genes_by_cell.csv">六基因表达数据</a></li>
<li><a href="data/processed/normalization_audit_212_values.csv">212 项归一化校验</a> / <a href="data/processed/gene_probe_provenance.csv">逐探针来源</a></li>
<li><a href="results/protocol.json">实验配置与假设</a> / <a href="results/verification.json">结果校验记录</a> / <a href="README.md">运行说明</a></li></ul>
<footer>论文：A machine learning approach to predict cellular uptake of pBAE polyplexes, DOI 10.1039/D3BM00741C。<a href="https://dau.url.edu/bitstream/handle/20.500.14342/4643/Biomat_scie_2023-11_a-machine.pdf?sequence=1">论文 PDF</a> · <a href="https://pubs.rsc.org/en/content/articlehtml/2023/bm/d3bm00741c">期刊页面</a> · <a href="https://github.com/mbhaylett23/pBAE-cellular-uptake-ML">作者仓库</a>。作者仓库固定提交：15e8e34a062f023cb1844fe5762d32f459bc1ef0。本报告为独立复现，未联系作者。<br>复现项目：<a href="{REPOSITORY_URL}">{REPOSITORY_URL}</a></footer>
</main></body></html>'''
(ROOT/'report.html').write_text(body,encoding='utf-8')
print(ROOT/'report.html')
