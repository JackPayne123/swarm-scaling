import json,glob,os,re,zipfile,statistics,collections,sys
def text_of(f):
    e=os.path.splitext(f)[1].lower()
    try:
        if e in ('.eml','.txt','.json'):
            return open(f,errors='ignore').read()
        z=zipfile.ZipFile(f)
        if e=='.docx':
            x=z.read('word/document.xml').decode('utf8','ignore')
            x=re.sub(r'</w:p>','\n',x); return re.sub(r'<[^>]+>','',x)
        if e=='.xlsx':
            out=''
            for n in z.namelist():
                if n.startswith('xl/sharedStrings') or n.startswith('xl/worksheets/sheet'):
                    out+=re.sub(r'<[^>]+>',' ',z.read(n).decode('utf8','ignore'))
            return out
        if e=='.pptx':
            out=''
            for n in z.namelist():
                if n.startswith('ppt/slides/slide'): out+=re.sub(r'<[^>]+>',' ',z.read(n).decode('utf8','ignore'))
            return out
    except Exception as ex:
        return ''
    return ''
cache={}
def dir_stats(d):
    if d in cache: return cache[d]
    fs=[f for f in glob.glob(d+'/**/*',recursive=True) if os.path.isfile(f)]
    chars=sum(len(text_of(f)) for f in fs)
    cache[d]=(len(fs),sum(os.path.getsize(f) for f in fs),chars); return cache[d]
rows=[]
for t in glob.glob('tasks/**/task.json',recursive=True):
    d=json.load(open(t)); base=os.path.dirname(t)
    dd=os.path.normpath(os.path.join(base,d['docs_dir'])) if d.get('docs_dir') else base+'/documents'
    n,b,c=dir_stats(dd) if os.path.exists(dd) else (0,0,0)
    parts=t.split(os.sep)
    rows.append(dict(area=parts[1],id='/'.join(parts[1:-1]),n_docs=n,bytes=b,chars=c,n_crit=len(d['criteria']),wt=d.get('work_type'),shared=bool(d.get('docs_dir'))))
json.dump(rows,open('../lab_rows.json','w'))
print(len(rows))
