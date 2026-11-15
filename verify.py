from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import json,hashlib
import pandas as pd
import numpy as np
from pipeline import features,prepare
root=Path(__file__).resolve().parent
raw=pd.read_csv(root/'data/metropt3.csv');rows=[]
for i,(a,b) in enumerate([(0,910168)]+[(910168+j*151695,910168+(j+1)*151695) for j in range(4)]):
 p=root/'data'/('initial.csv' if i==0 else f'batch_{i:02}.csv');part=raw.iloc[a:b];part.to_csv(p,index=False)
 rows.append({'file':p.name,'rows':len(part),'start':part.timestamp.iloc[0],'end':part.timestamp.iloc[-1],'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(root/'manifest.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
small=raw.iloc[:20000];a=features(small.iloc[:19000]);b=features(small)
safe=pd.Timestamp(small.timestamp.iloc[18999]).floor('10s');pd.testing.assert_frame_equal(a.loc[:safe],b.loc[a.index[a.index<=safe]],check_exact=True)
seam=10000;old=root/'data/check_old.csv';new=root/'data/check_new.csv';small.iloc[:seam].to_csv(old,index=False);small.iloc[seam:].to_csv(new,index=False)
chunk=prepare(new,old);reference=b.loc[chunk.index]
np.testing.assert_allclose(chunk.to_numpy(),reference.to_numpy(),atol=1e-8,rtol=1e-8)
out={'future_invariance_exact':True,'batch_context_allclose':True,'atol':1e-8,'rtol':1e-8,'checked_batch_rows':len(chunk)}
(root/'results/verification.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(out)
