"""Strict field parsers and untruncated, padding-aware generation."""
import re
from decimal import Decimal,InvalidOperation
PARSER_VERSION="strict-fields-v3"
NUMBER=r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?"

def normalize_answer(text):return " ".join(str(text).strip().casefold().split())

def numeric_answer(text):
    text=str(text).strip().replace(",","").replace("−","-").strip(" $\\")
    text=text[:-1] if text.endswith(".") else text
    try:
        value=Decimal(text)
        return ("0" if value==0 else format(value.normalize(),"f")) if value.is_finite() else None
    except InvalidOperation:return None

def output_region(response,mode="direct"):
    if mode=="scratchpad":return response.rsplit("</scratchpad>",1)[1] if "</scratchpad>" in response else ""
    return response

def answer_field(response,mode="direct"):
    text=output_region(response,mode).replace("**","")
    fields=re.findall(r"(?im)^[ \t]*(?:final[ \t]+)?answer[ \t]*:[ \t]*([^\n]+)",text)
    if fields:return re.split(r"(?i)\bconfidence\s*:",fields[-1])[0].strip()
    marked=re.findall(r"(?m)^\s*####\s*([^\n]+)",text)
    if marked:return marked[-1].strip()
    lines=[x.strip() for x in text.splitlines() if x.strip() and not re.match(r"(?i)^confidence\s*:",x.strip())]
    return lines[0] if len(lines)==1 and len(lines[0])<=80 else None

def parse_dataset_answer(response,dataset,mode="direct",choices=None,subtask=None):
    field=answer_field(response,mode)
    if field is None:return None
    if dataset=="gsm8k":
        m=re.fullmatch(rf"\s*\$?({NUMBER})\s*(?:[.!]|\$)?\s*",field)
        return numeric_answer(m.group(1)) if m else None
    if dataset in ("truthfulqa","arc","arc_challenge") or (dataset=="bbh" and subtask!="causal_judgement"):
        m=re.fullmatch(r"\s*\(?([A-Z])\)?[.!]?\s*",field,re.I)
        if m:
            label=m.group(1).upper()
            return label if choices is None or ord(label)-65<len(choices) else None
        if dataset!="bbh":return None
    if dataset=="bbh":
        value=normalize_answer(field).strip(".!()")
        return value if value in ("yes","no") else None
    return None

def parse_gsm8k_answer(response):return parse_dataset_answer(response,"gsm8k")
def parse_bbh_answer(response):return parse_dataset_answer(response,"bbh")
def parse_truthfulqa_answer(response):return parse_dataset_answer(response,"truthfulqa")
def parse_arc_answer(response):return parse_dataset_answer(response,"arc")

def parse_confidence(response,mode="direct"):
    text=output_region(response,mode).replace("**","")
    fields=re.findall(r"(?im)^[ \t]*confidence[ \t]*:[ \t]*([^\n]+)",text)
    if not fields:return None
    m=re.fullmatch(r"\s*([+-]?\d+(?:\.\d+)?)\s*%?\s*",fields[-1])
    if not m:return None
    value=float(m.group(1));return value/100 if 0<=value<=100 else None

def check_correctness(predicted,gold,dataset):
    if predicted is None:return False
    if dataset=="gsm8k":
        value=numeric_answer(predicted)
        return value is not None and value==numeric_answer(gold)
    return normalize_answer(predicted).strip("()") == normalize_answer(gold).strip("()")

def format_messages(messages,tokenizer):
    if "gemma" in tokenizer.name_or_path.lower() and messages[0]["role"]=="system":
        messages=[dict(role="user",content=messages[0]["content"]+"\n\n"+messages[1]["content"])]+messages[2:]
    return tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)

def generate_batch(model,tokenizer,prompts,max_new_tokens=2048,sample=False,seed=42):
    import torch
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    formatted=[format_messages(p,tokenizer) for p in prompts]
    enc=tokenizer(formatted,return_tensors="pt",padding=True,truncation=False).to(model.device)
    width=enc.input_ids.shape[1];limit=getattr(model.config,"max_position_embeddings",None)
    if limit and width+max_new_tokens>limit:raise ValueError(f"Context overflow: {width}+{max_new_tokens}>{limit}; no input truncated")
    kwargs=dict(max_new_tokens=max_new_tokens,do_sample=sample,pad_token_id=tokenizer.pad_token_id,use_cache=True)
    if sample:kwargs.update(temperature=0.7,top_p=0.95)
    with torch.inference_mode():out=model.generate(**enc,**kwargs)
    eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos]);records=[]
    for i,seq in enumerate(out):
        ids=seq[width:].tolist();end=next((j for j,t in enumerate(ids) if t in eos),len(ids));content_ids=ids[:end]
        prompt_ids=enc.input_ids[i][enc.attention_mask[i].bool()].tolist()
        records.append(dict(raw_response=tokenizer.decode(content_ids,skip_special_tokens=True),prompt_token_ids=prompt_ids,generated_token_ids=content_ids,input_tokens=len(prompt_ids),generated_tokens=end,stopped_eos=end<len(ids),hit_token_cap=end==max_new_tokens))
    return records

def generate_responses(model,tokenizer,prompts,max_new_tokens=2048,batch_size=4):
    return [r["raw_response"] for start in range(0,len(prompts),batch_size) for r in generate_batch(model,tokenizer,prompts[start:start+batch_size],max_new_tokens)]
