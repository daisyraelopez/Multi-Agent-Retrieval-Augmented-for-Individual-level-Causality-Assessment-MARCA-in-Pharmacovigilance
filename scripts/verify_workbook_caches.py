"""Read-only XLSX/CSV integrity check for the frozen KB (no spreadsheet edits)."""
from pathlib import Path
from collections import defaultdict
import zipfile, re, csv
ROOT=Path(__file__).resolve().parents[1]
kb=ROOT/'data/knowledge_base/expert_kb.xlsx'
with zipfile.ZipFile(kb) as z:
    if z.testzip() is not None:raise ValueError('Corrupt XLSX archive')
    xml=z.read('xl/worksheets/sheet1.xml').decode('utf-8')
rx=re.compile(r'<c\b[^>]*\br="([A-Z]+)(\d+)"[^>]*?(?<!/)>(.*?)</c>',re.S)
rows=defaultdict(dict)
for match in rx.finditer(xml):
    col,row,inside=match.group(1),int(match.group(2)),match.group(3)
    if row<=1:continue
    val=re.search(r'<v>(.*?)</v>',inside,re.S)
    if val: rows[row][col]=val.group(1)
assert len(rows)==1562,len(rows)
ai=['E','J','O','T','Y','AD','AI','AN','AS','AX']
hu=['G','L','Q','V','AA','AF','AK','AP','AU','AZ']
ag=['I','N','S','X','AC','AH','AM','AR','AW','BB']
with (ROOT/'output/results/predictions_rag.csv').open(newline='') as f:pred={int(x['dec_id']):x for x in csv.DictReader(f)}
with (ROOT/'data/gold_standard/gold_standard.csv').open(newline='') as f:gold={int(x['dec_id']):x for x in csv.DictReader(f)}
for row,d in rows.items():
    dec=int(d['A']);assert dec==row-1
    a=[int(d[x]) for x in ai]; h=[int(d[x]) for x in hu]
    for idx,q in enumerate(range(1,11)):
        assert a[idx]==int(pred[dec][f'Q{q}']),(dec,q,'MARCA')
        assert h[idx]==int(gold[dec][f'Q{q}']),(dec,q,'expert')
        assert d[ag[idx]]==('Agree' if a[idx]==h[idx] else 'Disagree'),(dec,q,'agreement cache')
    assert int(d['BC'])==sum(a),(dec,'ai total')
    assert int(d['BD'])==sum(h),(dec,'human total')
    assert d['BE']==('Agree' if sum(a)==sum(h) else 'Disagree'),(dec,'total agreement')
print(f'PASS: {len(rows)} KB records: all question scores, totals and agreement caches match frozen CSV inputs.')
