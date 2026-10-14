"""Versioned prompts shared by every experiment. No conflicting instructions."""
PROMPT_VERSION="explicit-fields-v3"
SYSTEM_DIRECT="Answer the question without an explanation."
SYSTEM_COT="Reason step by step, then give the final answer."
CONFIDENCE_SUFFIX="Report your probability that the final answer is correct as a percentage from 0 to 100."
def question_text(question,choices=None):
    return question+("\nChoices:\n"+"\n".join(f"({chr(65+i)}) {c}" for i,c in enumerate(choices)) if choices else "")
def make_prompt(question,mode,dataset,choices=None,confidence=False,trace=None):
    instructions=SYSTEM_COT if mode in ("cot","scratchpad") else SYSTEM_DIRECT
    if mode=="scratchpad":instructions="Put reasoning inside <scratchpad>...</scratchpad>, then give the final answer outside the tags."
    if trace is not None:instructions="Use the provided text as context, then answer the original question. Do not repeat the provided text."
    answer_format="a numeric value only" if dataset=="gsm8k" else ("a choice letter only" if choices else "the requested short answer only")
    instructions+=f"\nEnd with a separate line: Answer: <{answer_format}>."
    if confidence:instructions+="\n"+CONFIDENCE_SUFFIX+"\nAfter Answer, write a separate line: Confidence: <percentage>."
    body="Question:\n"+question_text(question,choices)
    if trace is not None:body+="\n\nProvided text:\n"+trace
    return [dict(role="system",content=instructions),dict(role="user",content=body)]
def make_accuracy_prompt(question,mode,dataset,choices=None):return make_prompt(question,mode,dataset,choices)
def make_calibration_prompt(question,mode,dataset,choices=None):return make_prompt(question,mode,dataset,choices,confidence=True)
make_consistency_prompt=make_accuracy_prompt
def make_paraphrase_gen_prompt(question):
    return [dict(role="system",content="Rewrite the question in five different ways. Preserve every number, entity, condition, negation, unit and requested quantity. Do not solve it or add facts. Return only a JSON array of five complete question strings."),dict(role="user",content=question)]
