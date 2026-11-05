from pathlib import Path
import subprocess,sys
r=Path.cwd();b=r/'mower_obstrast_check/runs';run=b/'D10_D02_hardaug_ft60';out=run/'analysis_20261104';out.mkdir(exist_ok=True);(out/'logs').mkdir(exist_ok=True)
scopes=[('test140',r/'mower_obstrast_check/data/test_sets/synthetic_8cls_140_20261030/all140.yaml','test'),('test90',r/'mower_obstrast_check/data/test_sets/synthetic_8cls_140_20261030/base90.yaml','test'),('test50',r/'mower_obstrast_check/data/test_sets/synthetic_8cls_140_20261030/mixed50.yaml','test'),('full',r/'mower_obstrast_check/data/yolo_check/data.yaml','val'),('old',r/'.p2-stage/d07_old.yaml','val')]
for scope,data,split in scopes:
 print('START',scope,flush=True)
 with (out/'logs'/f'{scope}.log').open('w',encoding='utf-8') as log:
  subprocess.run([sys.executable,'-X','utf8',str(r/'.p2-stage/evaluate_detect_d00.py'),'--weights',str(run/'weights/best.pt'),'--data',str(data),'--split',split,'--out',str(b/f'd10{scope}_20261104')],stdout=log,stderr=subprocess.STDOUT,check=True)
 print('DONE',scope,flush=True)
