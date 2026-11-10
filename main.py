"""Vibration-based bearing fault diagnosis with recording-disjoint load holdout."""
from pathlib import Path
import json,hashlib,platform
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import kurtosis,skew
from scipy.signal import welch
import scipy,sklearn,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,classification_report,confusion_matrix,ConfusionMatrixDisplay

FS=12000; WINDOW=2048; SEED=42
LABELS=['ball','inner_race','outer_race']

def features(raw):
    x=raw-np.mean(raw); rms=np.sqrt(np.mean(x*x)); eps=1e-12
    f,p=welch(x,fs=FS,nperseg=512)
    total=np.sum(p)+eps
    out={'rms':rms,'std':np.std(x),'peak_to_peak':np.ptp(x),'skewness':skew(x),
      'kurtosis_pearson':kurtosis(x,fisher=False),'crest_factor':np.max(abs(x))/(rms+eps),
      'shape_factor':rms/(np.mean(abs(x))+eps),'spectral_centroid_hz':np.sum(f*p)/total}
    for low,high in [(0,1000),(1000,2000),(2000,3000),(3000,4000),(4000,6001)]:
        out[f'power_fraction_{low}_{min(high,6000)}']=np.sum(p[(f>=low)&(f<high)])/total
    if not np.isfinite(list(out.values())).all(): raise ValueError('Non-finite features')
    return out

def evaluate(y,pred):
    return {'accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
      'macro_f1':float(f1_score(y,pred,labels=LABELS,average='macro',zero_division=0))}

def main():
    manifest=json.loads(Path('data/manifest.json').read_text())
    if len(manifest)!=36: raise ValueError('Run download_data.py for all 36 recordings')
    out=Path('results'); out.mkdir(exist_ok=True)
    rows=[]; examples={}
    for rec in manifest:
        path=Path('data')/rec['file_name']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=rec['sha256']: raise ValueError(f'Hash mismatch: {path}')
        x=loadmat(path)[rec['channel']].ravel().astype(float)
        if rec['sample_rate_hz']!=FS: raise ValueError('Unexpected sample rate')
        split='train' if rec['load_hp'] in [0,1] else ('validation' if rec['load_hp']==2 else 'test')
        for start in range(0,len(x)-WINDOW+1,WINDOW):
            raw=x[start:start+WINDOW]
            rows.append(dict(recording=rec['file_name'],start_sample=start,split=split,label=rec['label'],load_hp=rec['load_hp'],fault_inches=rec['fault_inches'],**features(raw)))
        examples.setdefault(rec['label'],x[:WINDOW])
    df=pd.DataFrame(rows)
    assert len(set(df[df.split=='train'].recording)&set(df[df.split=='test'].recording))==0
    assert df.groupby('recording').split.nunique().max()==1
    feature_names=list(features(np.random.default_rng(0).normal(size=WINDOW)))
    train=df[df.split=='train']; val=df[df.split=='validation']; test=df[df.split=='test']
    for part in [train,val,test]: assert set(part.label)==set(LABELS)
    models={'majority_baseline':DummyClassifier(strategy='most_frequent'),
      'svm_rbf':make_pipeline(StandardScaler(),SVC(C=10,gamma='scale',class_weight='balanced')),
      'random_forest':RandomForestClassifier(n_estimators=200,min_samples_leaf=2,class_weight='balanced',random_state=SEED,n_jobs=-1)}
    validation={}
    for name,model in models.items():
        model.fit(train[feature_names],train.label)
        validation[name]=evaluate(val.label,model.predict(val[feature_names]))
    winner=max(validation,key=lambda k:validation[k]['macro_f1'])
    final=models[winner].fit(pd.concat([train,val])[feature_names],pd.concat([train,val]).label)
    pred=final.predict(test[feature_names])
    report=evaluate(test.label,pred)
    metrics={'validation':validation,'selected_model':winner,'test':report,'seed':SEED,'window_samples':WINDOW,
      'sample_rate_hz':FS,'overlap_samples':0,'feature_names':feature_names,'classes':LABELS,
      'split':'train loads 0/1 hp; validation 2 hp; final refit 0/1/2 hp; test 3 hp',
      'window_counts':df.groupby('split').size().to_dict(),'recording_counts':df.groupby('split').recording.nunique().to_dict(),
      'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'pandas':pd.__version__,'scikit-learn':sklearn.__version__}}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    pd.DataFrame(classification_report(test.label,pred,labels=LABELS,output_dict=True,zero_division=0)).T.to_csv(out/'classification_report.csv')
    predictions=test[['recording','start_sample','label','load_hp','fault_inches']].copy();predictions['prediction']=pred
    predictions.to_csv(out/'test_predictions.csv',index=False)
    by_record=[]
    for name,part in predictions.groupby('recording'):
        by_record.append({'recording':name,'label':part.label.iloc[0],'majority_prediction':part.prediction.mode().iloc[0],
          'window_accuracy':float(np.mean(part.label==part.prediction)),'windows':len(part)})
    pd.DataFrame(by_record).to_csv(out/'per_recording_results.csv',index=False)
    df.to_csv(out/'window_features.csv',index=False)
    pd.DataFrame(manifest).assign(split=lambda d:d.load_hp.map({0:'train',1:'train',2:'validation',3:'test'})).to_csv(out/'data_manifest.csv',index=False)
    matrix=confusion_matrix(test.label,pred,labels=LABELS)
    pd.DataFrame(matrix,index=LABELS,columns=LABELS).to_csv(out/'confusion_matrix.csv')
    fig,ax=plt.subplots(figsize=(7,5));ConfusionMatrixDisplay(matrix,display_labels=LABELS).plot(ax=ax,cmap='Blues',colorbar=False)
    ax.set_title(f'Unseen-load test: {winner}');fig.tight_layout();fig.savefig(out/'confusion_matrix.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(10,8))
    for i,label in enumerate(LABELS):
        x=examples[label]; axes[i,0].plot(np.arange(len(x))/FS,x,linewidth=.6)
        axes[i,0].set(title=label,xlabel='Time (s)',ylabel='Recorded vibration amplitude')
        f,p=welch(x,fs=FS,nperseg=512);axes[i,1].semilogy(f,p+1e-15)
        axes[i,1].set(xlabel='Frequency (Hz)',ylabel='PSD (amplitude²/Hz)')
    fig.suptitle('Illustrative training recordings: 0.007 inch faults, 0 hp');fig.tight_layout();fig.savefig(out/'vibration_examples.png',dpi=150);plt.close(fig)
    print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
