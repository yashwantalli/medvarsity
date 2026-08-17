from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bertscore
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def bleu_score(reference, candidate):

    smmothie = SmoothingFunction().method1

    score = sentence_bleu(
        [reference.split()],
        candidate.split(),
        smoothing_function=smmothie
    )

    return score 

def rouge_scores(reference, candidate):
    
    scorer = rouge_scorer.RougeScorer(
        ["rouge1","rouge2","rougeL"],
        use_stemmer=True
    )

    scores = scorer.score(reference, candidate)

    return {
        "ROUGE-1": scores["rouge1"].fmeasure,
        "ROUGE-2": scores["rouge2"].fmeasure,
        "ROUGE-L": scores["rougeL"].fmeasure
    }

def meteor_scores(reference , candidate):
    score = meteor_score(
        [reference.split()],
        candidate.split()
    )

    return score


def bert_score(reference , candidate):
    P,R,F1 = bertscore(
        [candidate],
        [reference],
        lang="en",
        verbose = False
    )

    return {
        "precision": P.item(),
        "Recall": R.item(),
        "F1": F1.item()
    }

model= SentenceTransformer("all-MiniLM-L6-v2")
def semantic_similarity(text1 , text2):
    emb1= model.encode([text1])
    emb2=model.encode([text2])

    score = cosine_similarity(
        emb1,
        emb2
    )[0][0]

    return float(score)