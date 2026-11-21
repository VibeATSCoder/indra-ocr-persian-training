import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];work=ROOT/'v3/kernel-profile';work.mkdir(exist_ok=True)
code="import importlib.util,subprocess,sys\nfrom pathlib import Path\nmissing=[p for m,p in {'onnx':'onnx','pyarrow':'pyarrow'}.items() if importlib.util.find_spec(m) is None]\nif missing:subprocess.check_call([sys.executable,'-m','pip','install','-q',*missing])\np=Path('/tmp/ctc-profile-code');p.mkdir(exist_ok=True)\n"
for source in [ROOT/'indra_rtl_trainable.py',ROOT/'v3/profile_execution.py']:
    text=source.read_text();compile(text,source.name,'exec');code+=f"(p/{source.name!r}).write_text({text!r})\n"
code+="subprocess.check_call([sys.executable,'-u',str(p/'profile_execution.py')])\n"
(work/'run.py').write_text(code)
(work/'kernel-metadata.json').write_text(json.dumps({'id':'sharifi76/indra-persian-ctc-profile','title':'Indra Persian CTC Profile','code_file':'run.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':True,'enable_internet':True,'machine_shape':'NvidiaTeslaT4','dataset_sources':['sharifi76/persian-ocr-large-mixed-v3'],'kernel_sources':[],'competition_sources':[]}))
print('Built numerical performance probe')
