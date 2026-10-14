import itertools
import math
import numpy as np
import pytest
from src.metrics import *
from src.inference import *
from src.statistics import paired_metric,paired_randomization,holm
from src.protocol import paraphrase_screen

def test_endpoint_exactly_one_is_assigned():
    result=compute_ece([1.,1.,0.],[0,1,0])
    assert result["ece"]==pytest.approx(1/3)
    assert sum(b["bin_count"] for b in result["bin_data"])==3

@pytest.mark.parametrize("p,y",[([1.01],[1]),([-0.1],[0]),([np.nan],[0]),([.2],[.5]),([] ,[]),([.1,.2],[1])])
def test_invalid_metric_inputs_fail(p,y):
    with pytest.raises(ValueError):compute_ece(p,y)

def test_all_boundaries_and_binary_metric_constraints():
    rng=np.random.default_rng(3)
    for n in (10,100,1000):
        p=rng.choice(np.linspace(0,1,11),n);y=rng.integers(0,2,n)
        values=calibration_metrics(p,y)
        assert values["ece"]+1e-12>=abs(p.mean()-y.mean())
        assert values["brier"]<=1-y.mean()+1-p.mean()+1e-12
        assert values["brier"]==pytest.approx(values["reliability"]-values["resolution"]+values["uncertainty"])

def test_binned_murphy_is_not_reported_as_exact():
    p=[.91,.99];y=[0,1]
    exact=murphy_decomposition(p,y);binned=murphy_decomposition(p,y,10)
    assert abs(exact["residual"])<1e-12
    assert abs(binned["residual"])>.01
    assert binned["brier"]==pytest.approx(compute_brier(p,y))

def test_auroc_ties_and_undefined_class():
    assert compute_auroc([.8,.8,.8,.8],[0,1,0,1])==.5
    assert math.isnan(compute_auroc([.2,.3],[1,1]))

def test_aurc_ties_equal_average_over_permutations():
    y=np.array([0,1,0]);p=[.5]*3
    explicit=[]
    for order in itertools.permutations(range(3)):
        v=y[list(order)];explicit.append(np.mean([1-v[:k].mean() for k in range(1,4)]))
    assert compute_aurc(p,y)==pytest.approx(np.mean(explicit))

@pytest.mark.parametrize("text",["Answer: 98","There are 98 apples.","Confidence: -1","Confidence: 101","Confidence: 90 maybe","Confidence:\nAnswer: 80"])
def test_confidence_is_never_inferred_or_clipped(text):
    assert parse_confidence(text) is None

def test_confidence_decimal_and_one_hundred():
    assert parse_confidence("Answer: 2\nConfidence: 100%")==1
    assert parse_confidence("Confidence: 92.5")==.925

def test_answer_does_not_use_confidence_as_numeric_answer():
    assert parse_gsm8k_answer("Confidence: 98%") is None
    assert parse_gsm8k_answer("Reasoning: 4+5=9\nConfidence: 98%") is None
    assert parse_gsm8k_answer("Answer: 9\nConfidence: 98%")=="9"
    assert parse_gsm8k_answer("Answer:\nConfidence: 98%") is None

def test_numeric_sign_decimal_comma_punctuation():
    for text,gold in [("Answer: -0.5.","-0.5"),("Answer: 1,250","1250"),("Answer: 5.0","5"),("Answer: -0","0")]:
        assert check_correctness(parse_gsm8k_answer(text),gold,"gsm8k")
    assert not check_correctness("bad","bad","gsm8k")
    assert not check_correctness("-1","1","gsm8k")

def test_mcq_options_beyond_d_and_valid_option_range():
    assert parse_dataset_answer("Answer: (J)\nConfidence: 90","truthfulqa",choices=[str(i) for i in range(10)])=="J"
    assert parse_dataset_answer("Answer: J","truthfulqa",choices=["a","b"]) is None
    assert parse_bbh_answer("Answer: Yes")=="yes"
    assert parse_bbh_answer("Answer: (C)")=="C"

def test_scratchpad_must_close():
    assert parse_dataset_answer("<scratchpad>Answer: 4\nConfidence: 90","gsm8k","scratchpad") is None
    assert parse_dataset_answer("<scratchpad>reasoning</scratchpad>\nAnswer: 4","gsm8k","scratchpad")=="4"

def test_consistency_is_pairwise_and_no_singleton_or_missing_agreement():
    assert compute_consistency(["a","a","b"])==pytest.approx(1/3)
    assert math.isnan(compute_consistency(["a"]))
    assert compute_consistency([None,None])==0
    assert compute_consistency(["a",None,"a"])==pytest.approx(1/3)

def test_paired_bootstrap_preserves_identical_conditions():
    a=np.array([0,1,1,0]);result=paired_metric(a,a,np.mean,50)
    assert result["delta"]==result["ci_low"]==result["ci_high"]==0
    assert paired_randomization(a,a,np.mean,50)==1
    assert holm([.01,.02,np.nan])[0]==pytest.approx(.02)

def test_paraphrase_numeric_changes_rejected():
    assert paraphrase_screen("Alice has 12 apples. How many apples?","Alice has 13 apples. How many apples?")=="numeric_mismatch"
