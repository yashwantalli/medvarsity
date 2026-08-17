from pageindex import PageIndexClient
import json
from pathlib import Path
from litellm import completion
import time
from benchmark import benchmark
from typing import Any


WORKSPACE ="./workspace_gemma"
PDF_NAME = "attention-residuals.pdf"
PLANNER_MODEL = "ollama/qwen2.5-coder:7b"
ANSWER_MODEL = "ollama/qwen2.5-coder:7b"
RESULTS_DIR = Path("final_results")
COMBINED_RESULTS_PATH = RESULTS_DIR/"all_results.json"

MIN_SECTIONS = 5
MAX_SECTIONS = 8

client = PageIndexClient(workspace = WORKSPACE)


def find_document_id ( client:PageIndexClient , doc_name : str ) -> str:
    for doc_id , doc in client.documents.items():
        if doc.get("doc_name") == doc_name:
            return doc_id
    raise RuntimeError(f"Document '{doc_name}' not found in the given workspace")

def print_loaded_documents(client :PageIndexClient ) -> None:
    for did , doc in client.documents.items():
        print("id :",did)
        print("doc_name : ", doc.get("doc_name"))
        print()    

    print("="*60)
    print("loaded documents")
    print("="*60)

def load_document_structure(client :PageIndexClient, doc_id : str ) ->list[dict]:
    structure = json.loads(client.get_document_structure(doc_id))

    print(f"Top level sections : {len(structure)}")
    for node in structure:
        print("-",node["title"])
    return structure

def build_tree_text(structure:list[dict]) -> str:
    lines = []
    for node in structure :
        pages = f"{node["start_index"]}-{node["end_index"]}"
        summary = node.get("summary","")
        lines.append(
            f"node : {node["node_id"]}\n"
            f"title : -{node["title"]}\n" 
            f"pages  : {pages}\n"
            f"summary : {summary}\n\n"
        )
    return "\n".join(lines)


def build_planner_prompt(tree_text : str , question: str) -> str:
    return f"""
you are a retrieval planner. 

you are given ONLY the table of contents of a document.

your job is NOT to answer the user's question .

Your job is ONLY to decide which sections(s) should be retrieved.

if the answer is likely spread across multiple sections,
return multiple sections.

never return an empty listen unless the answer does not exist.

document structure:

{tree_text}

user question :
{question}

RETURN between {MIN_SECTIONS} and {MAX_SECTIONS} node_ids.

If multiple sections may contain useful information ,
include all of them.

Prefer returning more sections rather than fewer.

example :

{{
    "sections":[
        {{
            "node_id" : [
            "0002",
            "0003",
            "0004"
            ]
        }}
    ]
}}
"""


def clean_json_content(raw_content: str) -> str:
    content = raw_content.strip()
    content = content.replace("```json","").replace("```","")

    if content.lower().startswith("json"):
        content = content[4:].strip()

    start = content.find("{")
    end = content.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in LLM response: {raw_content!r}")

    return content[start:end+1]


def get_retrieval_plan(tree_text:str, question:str) -> dict:
    prompt = build_planner_prompt(tree_text,question)

    response = completion( 
        model = PLANNER_MODEL,
        messages = [
            {
                "role" : "system",
                "content" : "reply ONLY in valid JSON."
                },
            {
                "role":"user",
                "content": prompt
                },
            ],
            response_format = {"type": "json_object"},
            temperature=0,
    )

    raw_content = response.choices[0].message.content
    print("\nLLM Response:\n", repr(raw_content))

    cleaned = clean_json_content(raw_content)
    print("After cleaning:",repr(cleaned))

    return json.loads(cleaned)


def build_answer_prompt(context: str , question: str ) -> str:
    return f"""
you are answering questions about a research paper .

use only the provided context below . 

if the answer is not present, say " Not found in the retrieved pages " . 

context :
{context}

question :
{question}

give concise answer within 5-8 bullet points .

"""

def generate_answer(context:str , question: str) -> str:
    prompt = build_answer_prompt(context , question)
    response = completion(
        model = ANSWER_MODEL,
        message = [{"role" : "user", "content" : prompt}],
        temperature=0,
    )
    return response.choices[0].message.content

def retrieve_sections(client: PageIndexClient,doc_id: str,structure: list[dict],plan:dict,) -> tuple[list[dict],str]:

    nodes_by_id = {node["node_id"]: node for node in structure}

    retrieved_items: list[dict] = []
    retrieved_text= ""

    for section in plan["sections"]:
        for node_id in section["node_id"]:
            node = nodes_by_id[node_id]
            retrieved_items.append({
                "node_id" :node_id,
                "title" : node["title"],
                "strat_index" : node["start_index"],
                "end_index" : node["end_index"]
            })
            print ( "\n"+"="*60)
            print(node["title"])
            print("="*60)

            for page_num in range(node["start_index"], node["end_index"]+1):
                page = json.loads(
                    client.get_page_content(doc_id , str(page_num))
                )
                for p in page:
                    print(f"\n page {p['page']}")
                    print("-"*50)
                    print(p["content"][:200])
                    retrieved_text += p["content"] + "\n\n"

    unique_items = _dedupe_by_node_id(retrieved_items)
    return unique_items, retrieved_text

def _dedupe_by_node_id(items: list[dict]) -> list[dict]:
    seen = set()
    unique=[]
    for item in items:
        if item["node_id"] not in seen:
            seen.add(item["node_id"])
            unique.append(item)
    return unique



def run_question(client: PageIndexClient, doc_id:str, structure: list[dict], tree_text:str,qid:str,question:str) -> dict[str,Any]:
    print("\n"+ "="*80)
    print(f"running {qid}")
    print(question)
    print("="*80)
    total_start = time.time()
    retrieval_start = time.time()
    try:
        plan=get_retrieval_plan(tree_text,question)
    except Exception as e:
        print("\n ERROR:\n", tupe(e).__name__,e)
        raise
    retrieval_end=time.time()
    retrieved_items, retrieved_text = retrieve_sections(client, doc_id,structure,plan)
    print("\n Total retrieved characters: ", len(retrieved_text))

    generation_start = time.time()
    answer = generate_answer(retrieved_text, question)
    generation_end = time.time()

    total_end = time.time()

    retrieval_time = retrieval_end - retrieval_start
    generation_time = generation_end - generation_start
    total_time = total_end - total_start

    print(answer)
    print("\n" + "="*60)
    print("timings")
    print("="*60)
    print(f"retrieval time : {retrieval_time:.3f} sec")
    print(f"generation time : {generation_time:.3f} sec ")
    print(f"total time : {total_time:.3f} sec ")

    results = {
        "system": "PageIndex",

        "question": question,
    
        "top_k": len(retrieved_items),

        "retrieved_items" : retrieved_items,

        "retrieved_context" : retrieved_text,

        "answer" : answer,
    
        "retrieval_time" : retrieval_time,

        "generation_time" : generation_time,

        "pipeline_time" : total_time

    }


def main() -> None :
    client= PageIndexClient(workspace=WORKSPACE)

    print_loaded_documents(client)

    doc_id = find_document_id(client,PDF_NAME)
    print("document id : ", doc_id)

    print("\n"+"="*60)
    print("documents structure")    
    print("="*60)

    structure = load_document_structure(client, doc_id)

    print("\n"+"="*60)
    print("structure given to llm ")
    print("="*60)

    tree_text = build_tree_text(structure)

    RESULTS_DIR.mkdir(parents=True, exist_ok = True)
    all_results : dict[str,Any]={}

    for qid , item in benchmark.items():
        result = run_question(
            client = client,
            doc_id = doc_id,
            structure = structure,
            tree_text = tree_text,
            qid = qid,
            question = item["question"],
        )
        print("DEBUG result : ", repr(result))

        question_path = RESULTS_DIR / f"{qid}.json"
        with open (question_path, "w", encoding = "utf-8") as f:
            json.dump(result,f,indent = 4)
        print(f"\n Results saved to {question_path}")

        all_results[qid] = result 
        with open (COMBINED_RESULTS_PATH , "w" , encoding = "utf-8") as f:
            json.dump(all_results, f, indent =4)
        print(f"combined results updated at {COMBINED_RESULTS_PATH}")

if __name__ == "__main__":
    main()





