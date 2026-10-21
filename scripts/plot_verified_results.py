"""Publication-ready descriptive figures from saved CSV results (no API calls)."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
RES=ROOT/'output/results'; OUT=ROOT/'output/figures';OUT.mkdir(exist_ok=True,parents=True)
plt.rcParams.update({'figure.dpi':150,'savefig.dpi':250,'font.size':10,'axes.titlesize':13})

def save(name):
    plt.tight_layout()
    for ext in ('png','pdf'):
        plt.savefig(OUT/f'{name}.{ext}',bbox_inches='tight')
    plt.close()

q=pd.read_csv(RES/'question_level_agreement.csv')
sub=q[q.Question.isin([f'Q{i}' for i in range(2,11)])]
fig,ax=plt.subplots(figsize=(10,4.4))
ax.bar(sub.Question,sub['Agreement (%)'])
ax.set_ylim(0,110);ax.set_ylabel('Exact agreement (%)')
ax.set_title('Question-level MARCA–expert agreement (Q2–Q10)')
for x,y in enumerate(sub['Agreement (%)']):ax.text(x,y+1.2,f'{y:g}%',ha='center',fontsize=9)
save('Figure_question_level_agreement')

cm=pd.read_csv(RES/'confusion_matrix.csv',index_col=0)
fig,ax=plt.subplots(figsize=(6.8,5.3))
ax.imshow(cm.values,aspect='auto',cmap='Blues')
ax.set_xticks(range(len(cm.columns)),cm.columns)
ax.set_yticks(range(len(cm.index)),cm.index)
ax.set_xlabel('MARCA classification');ax.set_ylabel('Expert classification')
ax.set_title('Final causality classification — Naranjo Q1–Q10')
for i in range(cm.shape[0]):
 for j in range(cm.shape[1]):
  n=int(cm.iloc[i,j]);pct=100*n/cm.iloc[i].sum()
  ax.text(j,i,f'{n}\n({pct:.1f}%)',ha='center',va='center',color='white' if n>cm.values.max()*.5 else 'black')
save('Figure_final_classification_confusion_matrix')

ab=pd.read_csv(RES/'rag_ablation_agreement.csv')
fig,ax=plt.subplots(figsize=(7.5,4.4))
labels=['Q2','Q5','Q10'];x=np.arange(len(labels)); w=.32
non=[float(ab[(ab.Question==q)&(ab.Condition=='Non-RAG')]['Agreement (%)'].iloc[0]) for q in labels]
rag=[float(ab[(ab.Question==q)&(ab.Condition=='RAG')]['Agreement (%)'].iloc[0]) for q in labels]
ax.bar(x-w/2,non,w,label='Non-RAG');ax.bar(x+w/2,rag,w,label='Workbook-labelled RAG')
ax.set_xticks(x,labels);ax.set_ylim(0,110);ax.set_ylabel('Agreement (%)')
ax.set_title('Retrospective 60-DEC ablation (prediction provenance not verified)')
ax.legend(loc='lower center');save('Figure_rag_ablation')

d=pd.read_csv(RES/'directional_disagreement.csv');d=d[d.Question.isin(labels)]
fig,ax=plt.subplots(figsize=(8,4.4));x=np.arange(len(d))
ax.bar(x-.2,d['MARCA < Expert (n)'],.4,label='MARCA lower')
ax.bar(x+.2,d['MARCA > Expert (n)'],.4,label='MARCA higher')
ax.set_xticks(x,d.Question);ax.set_ylabel('Number of DECs')
ax.set_title('Directional scoring disagreements');ax.legend()
save('Figure_directional_disagreement')
print('Figures generated in',OUT)
