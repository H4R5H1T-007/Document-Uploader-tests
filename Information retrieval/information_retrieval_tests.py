import sys
sys.path.append('../')
import openai
import random
import os
import pandas as pd

from text_segmentation_model import Splitter
from info_retrieval_model import Retriever
import PyPDF2
import json

with open('information_retrieval_test_config.json') as f:
    test_config = json.load(f)

random.seed(0)
number_of_queries = test_config['num_of_queries']
number_of_results = test_config['num_of_res']
pdf_path = test_config['pdf_path']

def pdf_to_text(pdf_path):
    with open(pdf_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

doc_text = pdf_to_text(pdf_path)

splitter = Splitter(doc_text)
text_corpus = splitter.predict()
retriever = Retriever(text_corpus)

def query_creator(paragraph : str) -> str:
    # Generate the response using ChatGPT
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": "You are a helpful assistant whose job is to ask complex questions for which answers are from the given paragraph."},
                {"role": "user", "content": f"Can you please ask a complex question based on this paragraph without mentioning paragraph : {paragraph}"}
            ]
    )

    # Extract and return the generated queries
    query = response.choices[0].message.content
    return query

def query_bank_creator(text_chunks : list[str], number_of_samples : int = 10) -> list[list[str]]:
    query_bank = list()
    if number_of_samples > len(text_chunks):
        raise ValueError("number of samples cannot be greater than the length of the list.")
    
    file_path = f"./generated_questions/{str(splitter)}_{number_of_queries}.csv"
    if(not os.path.exists(file_path)):
        sampled_paragraphs = random.sample(text_chunks, number_of_samples)
        for paragraph in sampled_paragraphs:
            question = query_creator(paragraph)
            query_bank.append([paragraph, question])
        
        # paragraph = [x[0] for x in query_bank]
        # query = [x[1] for x in query_bank]

        # data = pd.DataFrame({key: value for query_bank in [paragraph, query] for key, value in zip(('paragraph', 'query'), query_bank)})
        data = pd.DataFrame(query_bank, columns=["paragraph", "query"])

        data.to_csv(file_path)

    else:
        print(f'Already generated questions are taken from file {file_path}')
        data = pd.read_csv(file_path, index_col=0)
        query_bank = [[x, y] for x, y in zip(data['paragraph'].to_list(), data['query'].to_list())]

        
    return query_bank

def test_suite_creator(
        chunks,
        number_of_samples=number_of_queries
    ):

    print(f"Now creating queries from randomly sampled {number_of_samples} samples using chatGPT api :-")
    queries_with_paragraphs = query_bank_creator(chunks, number_of_samples)
    # queries_with_paragraphs = [[paragraph, generated_query]]
    return queries_with_paragraphs


test_set = test_suite_creator(text_corpus, number_of_queries)


def text_id_generator(text_corpus):
    result = dict()
    for counter, text in enumerate(text_corpus):
        result[text] = f'd_{counter}'
    return result

def eval_dict_generator(id_dict, score, text_corpus):
    res = dict()
    for n, text in enumerate(text_corpus):
        res[id_dict[text]] = score[n]
    return res

def all_query_eval(test_set, id_dict, retriever):
    run_dict = dict()
    for query_id in range(len(test_set)):
        temp_res, predicted_score = retriever.retrieve(test_set[query_id][1], number_of_results)
        run_dict[f'q_{query_id}'] = eval_dict_generator(id_dict, predicted_score, temp_res)

    return run_dict

def qrels_dict_generator(test_set, id_dict):
    qrels_dict = dict()
    for query_id in range(len(test_set)):
        qrels_dict[f'q_{query_id}'] = {id_dict[test_set[query_id][0]] : 1}

    return qrels_dict

from ranx import evaluate, Qrels, Run
id_dict = text_id_generator(text_corpus)

qrels_dict = qrels_dict_generator(test_set, id_dict)
run_dict = all_query_eval(test_set, id_dict, retriever)

qrels = Qrels(qrels_dict)
run = Run(run_dict)
# print(qrels, run)

print(f"Evaluation metrics after using {str(retriever)} model for content retrieval and {str(splitter)} chunking method is")
print(evaluate(qrels, run, ["map@5", "mrr", "ndcg@5"]))