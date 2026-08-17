import json 
import csv
from pathlib import Path
from benchmark import benchmark
from metrics import bleu_score,rouge_scores, meteor_scores, bert_score,semantic_similarity


RESULTS_DIR = Path("final_results")
RESULTS_DIR.mkdir(parents=True, exist_ok = True)

def load_result(filename):
    with open(filename , "r", encoding = "utf-8") as f:
        return json.load(f)

def evaluate_system(result_json, reference):

    answer = result_json["answer"]
    question = benchmark["q1"]["question"]
    context = result_json["retrieved_context"]

    bleu = bleu_score(reference,answer)
    rouge= rouge_scores(reference,answer)
    meteor_val = meteor_scores(reference,answer)
    bert = bert_score(reference,answer)

    query_context = semantic_similarity(question , context)
    query_answer = semantic_similarity(question, answer)
    context_answer = semantic_similarity(context , answer)

    return {
        "BLEU": bleu,
        "ROUGE-1": rouge["ROUGE-1"],
        "ROUGE-2": rouge["ROUGE-2"],
        "ROUGE-L": rouge["ROUGE-L"],
        "meteor" : meteor_val,
        "BERT precision" : bert["precision"],
        "Bert Recall" : bert["Recall"],
        "BERT F1" : bert["F1"],
        "retrieval_time (s)": result_json["retrieval_time"],
        "generation_time (s)" : result_json["generation_time"],
        "total_time (s)" : result_json["pipeline_time"],
        "query-context": query_context,
        "query-answer" : query_answer,
        "context-answer": context_answer
    }

def metric_winner(metric , rag , pageindex):

    if "time" in metric.lower():
        if rag<pageindex:
            return "RAG"
        elif pageindex<rag:
            return "PageIndex"
        else:
            return "Tie"
    
    if rag>pageindex:
        return "RAG"
    elif pageindex>rag:
        return "PageIndex"
    else:
        return "Tie"

def main():
    rag_all =[]
    pageindex_all =[]
    # rows=[]

    for qid , item in benchmark.items():

        print("\n"+ "="*80)
        print(f"evaluating {qid}")
        print("="*80)
        reference = item["reference_answer"]

        rag = load_result(RESULTS_DIR / f"RAG_result_{qid}.json")
        pageindex = load_result(RESULTS_DIR / f"pageindex_result_{qid}.json")
        
        rag_scores = evaluate_system(rag , reference)
        pageindex_scores = evaluate_system(pageindex,reference)

        rag_all.append(rag_scores)
        pageindex_all.append(pageindex_scores)

        print(f"\n results for {qid}\n")

        print(f"{'metric':20}{'RAG':>15}{'PageIndex':>15}")
        print("-"*80)

        for metric in rag_scores:
            print(
                f"{metric:20}"
                f"{rag_scores[metric]:15.4f}"
                f"{pageindex_scores[metric]:15.4f}"
            )
        
    avg_rag = {}
    avg_pageindex = {}


    for metric in rag_all[0]:
        avg_rag[metric] = sum(
            result[metric] for result in rag_all
        )/len(rag_all)


        avg_pageindex[metric]= sum(
            result[metric] for result in pageindex_all
        )/len(pageindex_all)

    print("\n")
    print("="*70)
    print("avg results ")
    print("="*80)

    print(f"{'metric':20}{'RAG':>15}{'PageIndex':>15}")
    print("-"*80)

    for metric in avg_rag:
        print(
            f"{metric:20}"
            f"{avg_rag[metric]:15.4f}"
            f"{avg_pageindex[metric]:15.4f}"
        )

    question_ids = list(benchmark.keys())
    metric_names = list(avg_rag.keys())

    header = ["Metric"]



    for qid in question_ids:
        header.extend([
            f"{qid.upper()} RAG",
            f"{qid.upper()} PageIndex",
            f"{qid.upper()} Winner",
            f"{qid.upper()} Difference"
        ])

    header.extend([
        "Average RAG",
        "Average PageIndex",
        "Average Winner",
        "Average Difference"
    ])
    rows =[]

    rag_wins=0
    pageindex_wins=0
    ties=0

    for metric in metric_names :
        row= [metric]

        rag_values =[]
        page_values=[]

        for i , qid in enumerate(question_ids):
            rag=rag_all[i][metric]
            page= pageindex_all[i][metric]

            rag_values.append(rag)
            page_values.append(page)

            winner= metric_winner(metric,rag,page)

            if winner == "RAG":
                rag_wins += 1
            elif winner == "PageIndex":

                pageindex_wins += 1
            else:
                ties += 1
            
            difference = abs(rag-page)

            row.extend([
                round(rag,4),
                round(page,4),
                winner,
                round(difference,4)
            ])
        
        avgwinner = metric_winner(
            metric,
            avg_rag[metric],
            avg_pageindex[metric]
        )

        row.extend([
            round(avg_rag[metric],4),
            round(avg_pageindex[metric],4),
            avgwinner,
            round(avg_rag[metric]-avg_pageindex[metric],4)
        ])

        rows.append(row)

    outfile = RESULTS_DIR/ f"evaluation_results.csv"
    with open(outfile ,"w",newline="",encoding = "utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(header)

        writer.writerows(rows)

        writer.writerow([])

        writer.writerow(["overall summary"])

        writer.writerow(["RAG Wins",rag_wins])

        writer.writerow(["PageIndex Wins",pageindex_wins])

        writer.writerow(["Ties", ties])


    print(f"\n results saved to {outfile} ")


if __name__ == "__main__":

    main()
