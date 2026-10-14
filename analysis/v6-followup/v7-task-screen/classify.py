import ast, os, json, re, sys
R='ext/algotune/AlgoTuneTasks'
rows=json.load(open('alltasks.json'))
out={}
pats={
 'cpsat':r'cp_model|CpSolver|CpModel',
 'sat':r'pysat|from pysat|Glucose|Cadical|Minisat|RC2',
 'mip_lp_lib':r'pywraplp|pulp|pyomo|highspy|linprog|milp',
 'cvxpy':r'cvxpy|cp\.Problem|cp\.Variable',
 'cvx_int':r'boolean=True|integer=True',
 'fft':r'fft|dct|dst\b|idct|rfft',
 'conv':r'convolve|correlate|upfirdn|lfilter|fftconvolve',
 'ode':r'solve_ivp|odeint|diffrax',
 'nx':r'networkx|nx\.',
 'sklearn':r'sklearn',
 'linalg':r'np\.linalg|scipy\.linalg|la\.|linalg\.',
 'sparse':r'scipy\.sparse|sparse\.',
 'optimize':r'scipy\.optimize|optimize\.|minimize\(|leastsq|least_squares|root\(',
 'crypto':r'cryptography|hashlib|base64|gzip|zlib',
}
for r in rows:
    t=r['task']; p=f'{R}/{t}/{t}.py'
    if not os.path.exists(p): out[t]={'missing':True}; continue
    src=open(p).read()
    tree=ast.parse(src)
    cls=[n for n in tree.body if isinstance(n,ast.ClassDef)]
    solve=''
    for c in cls:
        for f in c.body:
            if isinstance(f,ast.FunctionDef) and f.name not in ('generate_problem','is_solution','__init__'):
                solve+=ast.unparse(f)+'\n'
    # include module-level funcs too
    for n in tree.body:
        if isinstance(n,ast.FunctionDef): solve+=ast.unparse(n)+'\n'
    imports='\n'.join(ast.unparse(n) for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom)))
    feats=[k for k,pt in pats.items() if re.search(pt, solve) or (k in ('cpsat','sat','cvxpy','nx','sklearn','crypto') and re.search(pt, imports) and re.search(pt.split('|')[0].split('\\')[0], solve))]
    out[t]={'feats':feats,'solve_lines':solve.count('\n')}
json.dump(out,open('classify.json','w'),indent=0)
for r in sorted(rows,key=lambda r:-r['best']):
    t=r['task']; o=out[t]
    print(f"{t[:38]:38s} {r['best']:8.2f} {r['median']:6.2f} {','.join(o.get('feats',[]))}  L={o.get('solve_lines')}")
