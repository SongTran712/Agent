from elasticsearch import Elasticsearch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from geneate_qa.classDocling import DoclingProcessor
from geneate_qa.classSummarize import SummarizeText
from geneate_qa.classDefineCT import VerilogProcessor
from geneate_qa.classGenQA import QADatasetGenerator
import pandas as pd

es = Elasticsearch("http://localhost:9200")
client = es.info()
print(client)
# ---------------------
from sentence_transformers import SentenceTransformer
# # ---------------------
def ProcessPDF(pdf, model): # NOTE: nếu có host_id thì thêm ở đây và các class!
    processor = DoclingProcessor(source=pdf)
    sections = processor.process_document()
    a = str(pdf)+": "
    store = []
    for i in range(len(sections)):
      if len(a) <= 3000:
        # print(i)
        a = a + sections[i] + "\n"
      else:
        print(i)
        store.append(a)
        a = str(pdf)+": "
        a = a + sections[i] + "\n"
    store.append(a)
    refine = VerilogProcessor(model_id = model, host = None)
    refine_text = [refine.format_code(sto) for sto in store]
    model_embed = SentenceTransformer('all-mpnet-base-v2')
    model_sum = SummarizeText(model_id = model, host = None)
    summary = [model_sum.generate_dataset(content) for content in refine_text]
    summary_embed = [model_embed.encode(content) for content in summary]
    # -------------------
    sum_data = []
    for i in range(len(refine_text)):
        sum_data.append({
            "content": refine_text[i],
            "summary": summary[i],
            "summary_embed": summary_embed[i]
        })
    generator = QADatasetGenerator(model_id = model, host = None)
    dataQA = [generator.generate_dataset(ref) for ref in refine_text]
    # # --------------------
    # Flatten the list of lists
    flattened_dataQA = []
    for sublist in dataQA:
        flattened_dataQA.extend(sublist)
    # Create a Pandas DataFrame
    QAdata = pd.DataFrame(flattened_dataQA)
    # # -------------
    return sum_data, QAdata, pdf
Processed_data, Generate_QA, pdf = ProcessPDF("code.pdf", "llama3-70b-8129")
from indexMapping import indexMapping
index_name = 'data'
if es.indices.exists(index=index_name):
  es.indices.delete(index=index_name)
  print(f"Index '{index_name}' deleted.")
es.indices.create(index=index_name, mappings=indexMapping)
# --------
for item in Processed_data:
    try:
      es.index(index=index_name, document=item)
    except Exception as e:
        print(f"Error indexing document: {e}")
        print(f"Document content: {item}")
# ---
