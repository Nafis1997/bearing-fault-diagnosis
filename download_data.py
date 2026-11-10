"""Download a fixed, documented CWRU subset from the university's official host."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,re,urllib.request,time
from scipy.io import loadmat
BASE='https://engineering.case.edu'
PAGE=BASE+'/bearingdatacenter/12k-drive-end-bearing-fault-data'

def download(rec):
    path=Path('data')/rec['file_name']
    for attempt in range(3):
        try:
            if not path.exists():
                with urllib.request.urlopen(rec['url'],timeout=90) as response: blob=response.read()
                # Check bytes before committing; HTML error pages are not MAT data.
                from io import BytesIO
                mat=loadmat(BytesIO(blob))
                if not any(k.endswith('_DE_time') for k in mat): raise ValueError('No drive-end signal')
                path.write_bytes(blob)
            mat=loadmat(path)
            keys=[k for k in mat if k.endswith('_DE_time')]
            if len(keys)!=1: raise ValueError(f'Expected one DE channel: {keys}')
            rec.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),channel=keys[0],samples=int(mat[keys[0]].size))
            print('Ready:',path.name,flush=True)
            return rec
        except Exception:
            if attempt==2: raise
            time.sleep(1+attempt)

def main():
    Path('data').mkdir(exist_ok=True)
    with urllib.request.urlopen(PAGE,timeout=90) as response: html=response.read().decode()
    records=[]
    for href,label in re.findall(r'<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>',html,re.S):
        label=re.sub('<[^>]+>','',label).strip()
        match=re.fullmatch(r'(IR|B|OR)(007|014|021)(?:@6)?_([0-3])',label)
        if not match: continue
        # Only the @6 location for outer race; avoid mixing locations.
        if match[1]=='OR' and '@6' not in label: continue
        url=href if href.startswith('https://') else BASE+href
        if not url.startswith(BASE+'/sites/default/files/'): raise ValueError('Unexpected data host')
        records.append(dict(file_name=url.rsplit('/',1)[-1],label={'IR':'inner_race','B':'ball','OR':'outer_race'}[match[1]],
          fault_inches=int(match[2])/1000,load_hp=int(match[3]),source_label=label,url=url,sample_rate_hz=12000,source_page=PAGE))
    if len(records)!=36 or len(set(r['file_name'] for r in records))!=36:
        raise ValueError('Expected 36 unique recordings. Source page may have changed; inspect it before proceeding.')
    with ThreadPoolExecutor(max_workers=4) as pool: records=list(pool.map(download,records))
    records.sort(key=lambda r:(r['label'],r['fault_inches'],r['load_hp']))
    Path('data/manifest.json').write_text(json.dumps(records,indent=2))
    print('Saved data/manifest.json with provenance and SHA-256 hashes.')
if __name__=='__main__': main()
