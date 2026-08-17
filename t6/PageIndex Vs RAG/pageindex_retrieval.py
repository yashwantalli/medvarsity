from pageindex import PageIndexClient
import json
from pathlib import Path
from litellm import completion
import time
from benchmark import benchmark


WORKSPACE ="./workspace_gemma"
PDF_NAME = "attention-residuals.pdf"
client = PageIndexClient(workspace = WORKSPACE)
RESULTS_DIR = Path("final_results")
RESULTS_DIR.mkdir(parents=True, exist_ok = True)


for did , doc in client.documents.items():
    print("id :",did)
    print("doc_name : ", doc.get("doc_name"))
    print()    

print("="*60)
print("loaded documents")
print("="*60)


doc_id= next(
    (
        did
        for did , doc in client.documents.items()
        if doc.get("doc_name","").strip() == PDF_NAME

    ),
    None,
)

if doc_id is None:
    raise RuntimeError("document not in workspace")



print("document id :",doc_id)

print("\n"+"="*60)
print("documents metadata")
print("="*60)

metadata = json.loads(client.get_document(doc_id))


print("\n"+"="*60)
print("documents structure")
print("="*60)

structure = json.loads(client.get_document_structure(doc_id))


print(f"top level sections {len(structure)}")

for node in structure:
    print("-",node['title'])

for k,v in structure[0].items():
    print(k,":",type(v))

print("\n"+"="*60)
print("structure given to llm ")
print("="*60)

tree_text = ""
for node in structure :
    title = node["title"]
    pages = f"{node["start_index"]}-{node["end_index"]}"
    summary = node.get("summary","")
    tree_text += (
        f"node : {node["node_id"]}\n"
        f"title : -{title}\n" 
        f"pages  : {pages}\n"
        f"summary : {summary}\n\n")


for qid , item in benchmark.items():

    question = item["question"]

    print("\n"+ "="*80)
    print(f"running {qid}")
    print(question)
    print("="*80)

    total_start=time.time()


    prompt = f"""
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

    respond with the titles of the relevant sections.
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
    print("\n"+"="*60)
    print("prommpt sent to llm ")
    print("="*60)


    retrieval_start=time.time()
    try:
        response = completion( 
            model = "ollama/qwen2.5-coder:7b",
            messages = [
                {
                    "role" : "system",
                    "content" : "reply ONLY in valid JSON."
                },
                {
                    "role":"user",
                    "content": prompt
                }
            ],
            response_format = {"type": "json_object"},
            temperature=0
        )

        print("\n llm response \n")
        print(response.choices[0].message.content)

    except Exception as e:
        print("type of error")
        print(type(e).__name__)
        print(e)

    print(repr(response.choices[0].message.content))

    content = response.choices[0].message.content.strip()
    print(repr(content))

    content = content.replace("```json","")
    content = content.replace("```","")


    if content.lower().startswith("json"):
        content = content[4:].strip()

    start = content.find("{")
    end = content.rfind("}")

    content = content[start : end+1]

    print("after cleaning")
    print(repr(content))


    result= json.loads(content)
    retrieval_end=time.time()
    retrieval_time = retrieval_end - retrieval_start

    selected_nodes= {
        node["node_id"]: node
        for node in structure
    }


    retrieved_text = ""
    retrieved_items=[]
    for section in result["sections"]:
        for node_id in section["node_id"]:
            node = selected_nodes[node_id]
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

    seen = set()
    unique_items=[] 

    for item in retrieved_items :
        if item["node_id"] not in seen : 
            seen.add(item["node_id"])
            unique_items.append(item)

    retrieved_items = unique_items


    print('\n total length of retrieved characters : ',len(retrieved_text))

    prompt = f"""
    you are answering questions about a research paper .

    use only the provided context below . 

    if the answer is not present, say " Not found in the retrieved pages " . 

    context 
    {retrieved_text}

    question 
    {question}

    give concise answer within 5-8 bullet points .

    """

    generation_start = time.time()
    response = completion(
        model="ollama/qwen2.5-coder:7b",
        messages = [
            {
            "role" : "user",
            "content" : prompt
            }
        ],
        temperature = 0
    )

    answer =response.choices[0].message.content

    generation_end =time.time()
    generation_time = generation_end - generation_start

    print(answer)

    total_end=time.time()
    total_time = total_end - total_start

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
    output_path = RESULTS_DIR / f"pageindex_result_{qid}.json"

    with open(output_path,"w",encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"\n results saved to {output_path}")

