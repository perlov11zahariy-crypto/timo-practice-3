"""MetroPT-3: причинные признаки и воспроизводимые эксперименты ПР3."""
from pathlib import Path
import argparse, hashlib, json, time, platform, zipfile
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, recall_score
from sklearn.model_selection import GridSearchCV
import sklearn,joblib
ROOT=Path(__file__).resolve().parent
AN=['TP2','TP3','H1','DV_pressure','Reservoirs','Oil_temperature','Motor_current']
DI=['COMP','DV_eletric','Towers','MPG','LPS','Pressure_switch','Oil_level','Caudal_impulses']
NEW=['tp3_mean_30m','tp3_slope_10m','load_share_30m','load_switches_30m','pressure_delta']
EVENTS=[('2020-04-18','2020-04-18 23:59'),('2020-05-29 23:30','2020-05-30 06:00'),('2020-06-05 10:00','2020-06-07 14:30')]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(df):
    df=df.copy();df['timestamp']=pd.to_datetime(df['timestamp']);df=df.drop(columns=['Unnamed: 0'],errors='ignore').drop_duplicates()
    if df.timestamp.duplicated().any():raise ValueError('Конфликт временных меток')
    if not df.timestamp.is_monotonic_increasing:raise ValueError('Нарушен порядок времени')
    if not df[DI].isin([0,1]).all().all():raise ValueError('Недопустимый цифровой сигнал')
    if not np.isfinite(df[AN+DI].to_numpy()).all():raise ValueError('Пропуски/нечисловые значения: карантин')
    return df.set_index('timestamp')
def features(df):
    d=clean(df);groups=d.index.to_series().diff().gt(pd.Timedelta('60s')).cumsum();parts=[]
    for _,g in d.groupby(groups):
        r=g.resample('10s',closed='right',label='right').agg({**dict.fromkeys(AN,'median'),**dict.fromkeys(DI,'last')})
        seen=r.TP3.notna();r=r.ffill(limit=2)
        r[NEW[0]]=r.TP3.rolling(180,min_periods=180).mean()
        # На полной сетке OLS-наклон за 60 узлов; секунды, бар/с.
        t=pd.Series(np.arange(len(r),dtype=float)*10,index=r.index)
        r[NEW[1]]=r.TP3.rolling(60).cov(t)/t.rolling(60).var()
        r[NEW[2]]=r.DV_eletric.rolling(180,min_periods=180).mean()
        changes=r.DV_eletric.diff().abs();changes.iloc[0]=0
        r[NEW[3]]=changes.rolling(180,min_periods=180).sum()
        r[NEW[4]]=r.TP3-r.Reservoirs
        good=(seen.rolling(180,min_periods=180).mean()>=.8)&((r.index-r.index[0])>=pd.Timedelta('30min'))
        parts.append(r.loc[good & (r.index.second==0),AN+DI+NEW].dropna())
    return pd.concat(parts).sort_index()
def prepare(path,context=None):
    raw=pd.read_csv(path);start=pd.to_datetime(raw.timestamp).min()
    if context:
        old=pd.read_csv(context);old['timestamp']=pd.to_datetime(old.timestamp)
        old=old[(old.timestamp<start)&(old.timestamp>=start-pd.Timedelta('31min'))]
        raw=pd.concat([old,raw],ignore_index=True)
    return features(raw).loc[start:]
def make(model,cols):
    pre=ColumnTransformer([('numeric',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),cols)],remainder='drop')
    return Pipeline([('schema',FunctionTransformer()),('preprocess',pre),('model',clone(model))])
def label(x,end):
    y=pd.Series(0,index=x.index,dtype=int);valid=x.index<=end-pd.Timedelta('24h')
    for a,b in EVENTS:
        a,b=pd.Timestamp(a),pd.Timestamp(b)
        y.loc[(x.index>=a-pd.Timedelta('24h'))&(x.index<a)]=1
        valid &= ~((x.index>=a)&(x.index<=b))
    return y,valid
def train():
    result=ROOT/'results';result.mkdir(exist_ok=True)
    z=zipfile.ZipFile(ROOT/'data/metropt.zip');names=z.namelist();print(names,flush=True)
    name=next(n for n in names if n.lower().endswith('.csv'))
    rawpath=ROOT/'data/metropt3.csv'
    if not rawpath.exists():rawpath.write_bytes(z.read(name))
    assert sha(rawpath)=='db30ccb4ea402e3c8bf2c99db06e288d4f2a772f6928f9dbe26a920d69793e24','Изменился исходный CSV'
    raw=pd.read_csv(rawpath);assert len(raw)==1516948
    initial=raw.iloc[:910168].copy();initial.to_csv(ROOT/'data/initial.csv',index=False)
    x=features(initial);x2=features(initial);pd.testing.assert_frame_equal(x,x2,check_exact=True)
    x.to_csv(result/'features.csv',float_format='%.12g');y,valid=label(x,pd.to_datetime(initial.timestamp).max());x=x.loc[valid];y=y.loc[valid]
    fitmask=x.index<pd.Timestamp('2020-04-30');valmask=(x.index>=pd.Timestamp('2020-05-01'))&(x.index<pd.Timestamp('2020-05-31'))
    finalmask=x.index<pd.Timestamp('2020-05-31');testmask=x.index>=pd.Timestamp('2020-06-01')
    xt,yt=x.loc[fitmask],y.loc[fitmask];xv,yv=x.loc[valmask],y.loc[valmask]
    rows=[];models={}
    def run(name,m,cols=AN+DI+NEW,group='main'):
        p=make(m,cols);t=time.perf_counter();p.fit(xt,yt);secs=time.perf_counter()-t
        a=p.predict_proba(xt)[:,1];b=p.predict_proba(xv)[:,1]
        row=dict(name=name,group=group,train_ap=average_precision_score(yt,a),val_ap=average_precision_score(yv,b),roc_auc=roc_auc_score(yv,b),recall05=recall_score(yv,b>=.5,zero_division=0),fit_s=secs,params=str(m.get_params()))
        if isinstance(m,LogisticRegression):row['nonzero']=int(np.count_nonzero(p[-1].coef_))
        rows.append(row);models[name]=p;print(name,round(row['val_ap'],5),flush=True);pd.DataFrame(rows).to_csv(result/'experiments.csv',index=False)
        return p
    run('Dummy',DummyClassifier(strategy='prior'))
    for pen in ['l1','l2']:
        for c in [.01,.1,1,10]:run(f'LR_{pen}_{c}',LogisticRegression(penalty=pen,C=c,solver='liblinear',class_weight='balanced',random_state=42,max_iter=3000),group='regularization')
    run('GaussianNB',GaussianNB())
    for depth in [2,4,6,8,12,None]:run(f'Tree_{depth}',DecisionTreeClassifier(max_depth=depth,min_samples_leaf=30,class_weight='balanced',random_state=42),group='depth')
    run('RandomForest',RandomForestClassifier(n_estimators=100,max_depth=8,min_samples_leaf=30,class_weight='balanced',random_state=42,n_jobs=2))
    run('HistGradientBoosting',HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,l2_regularization=1,class_weight='balanced',early_stopping=False,random_state=42))
    # По одному настоящему временному разбиению: E1 в train, E2 в validation.
    xd=x.loc[fitmask|valmask];yd=y.loc[fitmask|valmask];splits=[(np.flatnonzero(xd.index<pd.Timestamp('2020-04-30')),np.flatnonzero(xd.index>=pd.Timestamp('2020-05-01')))]
    bases=sorted([r for r in rows if r['name']!='Dummy'],key=lambda r:-r['val_ap'])[:2]
    grids={}
    for r in bases:
        p=models[r['name']];m=p[-1]
        if isinstance(m,LogisticRegression):grid={'model__C':[.03,.1,.3,1,3]}
        elif isinstance(m,DecisionTreeClassifier):grid={'model__max_depth':[3,5,8],'model__min_samples_leaf':[20,60]}
        elif isinstance(m,RandomForestClassifier):grid={'model__max_depth':[5,10],'model__min_samples_leaf':[20,60]}
        elif isinstance(m,GaussianNB):grid={'model__var_smoothing':[1e-11,1e-9,1e-7]}
        else:grid={'model__learning_rate':[.03,.1],'model__max_leaf_nodes':[7,15]}
        gs=GridSearchCV(p,grid,scoring='average_precision',cv=splits,n_jobs=1,refit=False);gs.fit(xd,yd)
        pd.DataFrame(gs.cv_results_).to_csv(result/('grid_'+r['name']+'.csv'),index=False)
        m.set_params(**{k.split('__')[1]:v for k,v in gs.best_params_.items()});run(r['name']+'_tuned',m,group='tuned');grids[r['name']]={'grid':grid,'best':gs.best_params_,'ap':gs.best_score_}
    candidates=[r for r in rows if r['name']!='Dummy'];bestap=max(r['val_ap'] for r in candidates)
    eligible=[r for r in candidates if bestap-r['val_ap']<=.005]
    # Из сопоставимых по AP предпочитается линейная, затем NB, дерево, ансамбль.
    def rank(r):
        m=models[r['name']][-1]
        family=0 if isinstance(m,LogisticRegression) else 1 if isinstance(m,GaussianNB) else 2 if isinstance(m,DecisionTreeClassifier) else 3
        complexity=(m.max_depth or 999) if isinstance(m,DecisionTreeClassifier) else 0
        return (family,complexity,r['name'])
    winner=min(eligible,key=rank);chosen=models[winner['name']]
    run('Winner_without_new',chosen[-1],AN+DI,group='ablation')
    chosen.fit(x.loc[finalmask],y.loc[finalmask]);joblib.dump(chosen,result/'model_v1.joblib')
    pd.concat([x.loc[testmask],y.loc[testmask].rename('target')],axis=1).to_csv(result/'test_locked.csv')
    info={'winner':winner['name'],'best_ap':bestap,'selected':winner,'grids':grids,'raw_sha256':sha(rawpath),'features_sha256':sha(result/'features.csv'),'repeat_exact':True,'raw_rows':len(raw),'initial_rows':len(initial),'feature_rows':len(x2),'labelled_rows':len(x),'splits':{k:{'n':int(m.sum()),'positive':int(y.loc[m].sum()),'start':str(x.loc[m].index.min()),'end':str(x.loc[m].index.max())} for k,m in [('fit',fitmask),('validation',valmask),('final_train',finalmask),('test',testmask)]},'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__},'features':AN+DI+NEW,'seed':42,'test_evaluated':False}
    (result/'summary.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    lr=models['LR_l1_0.1'];pd.DataFrame({'feature':AN+DI+NEW,'coefficient':lr[-1].coef_[0]}).to_csv(result/'coefficients.csv',index=False)
    # Загрузка и повторное применение сохраненного артефакта.
    restored=joblib.load(result/'model_v1.joblib');np.testing.assert_array_equal(chosen.predict_proba(xv.iloc[:100]),restored.predict_proba(xv.iloc[:100]))
    print(json.dumps(info,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['train','prepare']);ap.add_argument('--input');ap.add_argument('--context');ap.add_argument('--output');a=ap.parse_args()
    if a.mode=='train':train()
    else:prepare(a.input,a.context).to_csv(a.output or ROOT/'results/prepared.csv')
