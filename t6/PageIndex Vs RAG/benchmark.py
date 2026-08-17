benchmark = {
    "q1" : {
        "question" : 
        "explain the complete workflow of block attention residuals from layer outputs to block representation and how this reduces memory and communication costs compared to full attention residuals. ",
        "reference_answer": 
        """
layers are partitioned into N blocks; within each block, ;layer outputs are summed into a single representation(intra-block accumulation).
accross blocks, each layer applies softmax attention over the N preceding block-level representations plus the token embedding, using a learned pseudo-query per layer, rather than attending to all L individual layer outputs.
this reduces memory/communication from O(Ld) to O(Nd), since only block summaries need to be cached and transmitted across pipeline stages.
a two-phase strategy batches inter-block attention seperately from sequential intra-block attention, emerged via online softmax. 
cross- stage caching avoids retransmitting blocks already received, cutting pipeline communication further.
"""
    },
    
    "q2" : {
        "question" :
        "what architectural trend emerges from the architecture sweep?",
        "reference_answer" :
        """
both baseline and attnres reach their lowest loss at H/Lb = 0.3,and loss decreases with growing dmodel/Lb and shrinking H/Lb in both cases.
attenres achieves lower loss than baseline in all 25 tested configurations, by 0.019-0.063.
the optimal dmodel/Lb shifts from = 60(baseline,loss 1.847)to = 45(attnres, loss 1.802).
a lower dmodel/Lb means a deeper, narrower network-so attenres exploits additional depth more efficiently than the baseline.
this depth preference doesnt directly imply a deployement recommendation,since deeper models incur higher latency due to sequential computation.
"""
    },

    "q3" :{
        "question" : 
        "which benchmark categories were used to evaluate the final models?",
        "reference_answer":
        """
general: MMLU,MMLU-Pro,GPQA-Diamond,BBH,ARC-Challenge,HellaSwag,TriviaQA-language understanding and reasoning.
math&code : GSM8K,MGSM,Math,CMath,HumanEval,MBPP-quntitative and code-generation reasoning.
chinese : CMMLU,C-Eval-chinese language understanding.
evaluationn followed the same protocol used for the kimi linear baseline model.
attnres matcher or outperformed the baseline across all benchmarks,with the largest gains on GPQA-Diamond(+7.5) and HumanEval(+3.1). 
"""

    }
}

