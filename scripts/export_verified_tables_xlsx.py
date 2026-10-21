"""Export a compact result workbook using artifact_tool. The KB itself is preserved separately."""
from pathlib import Path
import csv,json
from artifact_tool import Workbook, SpreadsheetFile
ROOT=Path(__file__).resolve().parents[1]
RES=ROOT/'output/results'
w=Workbook.create()

def sheet_from_rows(name,rows,widths=None):
    s=w.worksheets.add(name)
    n=len(rows[0]); assert all(len(r)==n for r in rows)
    s.get_range_by_indexes(0,0,len(rows),n).values=rows
    def col(n):
        ret=''
        while n:
            n,k=divmod(n-1,26);ret=chr(65+k)+ret
        return ret
    end=col(n)
    s.get_range(f'A1:{end}1').format={'fill':'#17446B','font':{'bold':True,'color':'#FFFFFF'},'row_height':26,'wrap_text':True}
    s.get_range_by_indexes(1,0,len(rows)-1,n).format.row_height=21
    for j in range(n):
        s.get_range(f'{col(j+1)}:{col(j+1)}').format.column_width=(widths[j] if widths else (35 if j==0 else 23))
    s.freeze_panes.freeze_rows(1)
    return s

def csvrows(filename):
    with (RES/filename).open(newline='',encoding='utf8') as f: rows=list(csv.reader(f))
    converted=[rows[0]]
    for row in rows[1:]:
        conv=[]
        for v in row:
            try:
                if v=='':conv.append(None)
                elif v.lstrip('-').isdigit(): conv.append(int(v))
                else: conv.append(float(v))
            except ValueError: conv.append(v)
        converted.append(conv)
    return converted

summary=json.loads((RES/'release_metrics.json').read_text())
rows=[['Verified metric','Value'],['DECs',summary['n_decs']],['ICSRs',summary['n_icsrs']],
      ['Final category agreement (%)',summary['final_agreement_pct']],['Cohen kappa',summary['final_unweighted_kappa']],
      ['Exact total-score agreement (%)',summary['exact_total_score_agreement_pct']],
      ['Mean total MARCA',summary['mean_total_marca']],['Mean total expert',summary['mean_total_human']],
      ['Mean MARCA − expert',summary['mean_signed_difference']],
      ['Q1 included in total', 'Yes'],['RAG provenance verified','No'],
      ['Correction','DEC 310 Q2 MARCA +1 to +2; rationale unchanged']]
sheet_from_rows('Overview',rows,[37,65])
for title,file,widths in [
    ('Question agreement','question_level_agreement.csv',[14,46,14,21,14,16,16]),
    ('Directional','directional_disagreement.csv',[13,22,22,22,20,14]),
    ('Classification','classification_level_agreement.csv',None),
    ('Confusion matrix','confusion_matrix.csv',[22,20,20,20]),
    ('RAG ablation','rag_ablation_agreement.csv',None),
]:
    sheet_from_rows(title,csvrows(file),widths)
out=RES/'MARCA_verified_results.xlsx'
SpreadsheetFile.export_xlsx(w).save(str(out))
print('RESULT WORKBOOK:',out,'bytes',out.stat().st_size)
print(w.inspect({'kind':'table','range':'Overview!A1:B12','include':'values','table_max_rows':12,'table_max_cols':2}).ndjson[:1500])
