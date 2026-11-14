"""Classify existing detector categories using a local text model, without DAM."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('HF_HUB_OFFLINE','1')
from utils.model_classification import parse_classification, cache_key

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('result_dir',type=Path)
parser.add_argument('--model-path',type=Path,default=Path('/media/joe/Data/Dynamic-GSG/models/Qwen2.5-1.5B-Instruct'))
parser.add_argument('--max-new-tokens',type=int,default=64)
args=parser.parse_args()
source=json.loads((args.result_dir/'scene_graph_geometric.json').read_text())
prompt_path=Path(__file__).resolve().parents[1]/'configs/prompts/parsing_objects_caption.txt'
prompt=prompt_path.read_text()+'\nReturn only the JSON object. Do not add explanation.'
manifest=json.loads((args.model_path/'download_manifest.json').read_text())
cache_path=args.result_dir/'model_type_cache.json'
cache=json.loads(cache_path.read_text()) if cache_path.exists() else {}
model=None;tokenizer=None
outputs={}
for category in sorted({n['category'] for n in source['nodes']}):
    key=cache_key(category,manifest['repo'],manifest['revision'],prompt)
    if key not in cache:
        if model is None:
            import torch
            from transformers import AutoModelForCausalLM,AutoTokenizer
            tokenizer=AutoTokenizer.from_pretrained(str(args.model_path),local_files_only=True)
            model=AutoModelForCausalLM.from_pretrained(str(args.model_path),local_files_only=True,torch_dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval()
            torch.cuda.reset_peak_memory_stats()
        messages=[dict(role='system',content=prompt),dict(role='user',content=f'Query object name: {category}')]
        text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=tokenizer(text,return_tensors='pt').to('cuda')
        with torch.inference_mode():
            generated=model.generate(**inputs,max_new_tokens=args.max_new_tokens,do_sample=False,pad_token_id=tokenizer.eos_token_id)
        raw=tokenizer.decode(generated[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True).strip()
        try:
            object_type=parse_classification(raw)
        except (ValueError,json.JSONDecodeError) as e:
            (args.result_dir/'model_classification_failure.json').write_text(json.dumps(dict(category=category,raw_response=raw,error=str(e)),indent=2))
            raise SystemExit(f'Invalid model JSON for {category!r}; saved failure for inspection. No rules fallback used.')
        cache[key]=dict(category=category,object_type=object_type,raw_response=raw,model=manifest['repo'],revision=manifest['revision'],prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),source='local_text_model')
        temporary=cache_path.with_suffix('.tmp');temporary.write_text(json.dumps(cache,indent=2));temporary.replace(cache_path)
    outputs[category]=cache[key]
    print(category,'->',outputs[category]['object_type'],flush=True)
result=dict(model=manifest,prompt_path=str(prompt_path),prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),dam_used=False,classification_input='YOLO category text only',generation=dict(do_sample=False,max_new_tokens=args.max_new_tokens,dtype='float16'),objects={str(n['id']):outputs[n['category']] for n in source['nodes']})
if model is None and (args.result_dir/'model_classifications.json').exists():
    previous=json.loads((args.result_dir/'model_classifications.json').read_text())
    if previous.get('model')==manifest and previous.get('prompt_sha256')==result['prompt_sha256']:
        for k in ['peak_cuda_allocated_gib','peak_cuda_reserved_gib']:
            if k in previous: result[k]=previous[k]
result['inference_ran_this_invocation']=model is not None
if model is not None:
    result['peak_cuda_allocated_gib']=torch.cuda.max_memory_allocated()/1024**3
    result['peak_cuda_reserved_gib']=torch.cuda.max_memory_reserved()/1024**3
(args.result_dir/'model_classifications.json').write_text(json.dumps(result,indent=2))
print('Saved model_classifications.json',flush=True)
