import torch
from sklearn.metrics.pairwise import cosine_similarity
from InstructorEmbedding import INSTRUCTOR
from abc import ABC, abstractclassmethod
import json
import os

import openai
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain import PromptTemplate, OpenAI, LLMChain

with open('../info_retrieval_config.json') as f:
    info_config = json.load(f)

openai.api_key = info_config['openai_key']  # Replace with your OpenAI API key
batch_size = info_config['batch_size']
huggingface_model = info_config['huggingface_model']
os.environ['OPENAI_API_KEY'] = openai.api_key

class RetrieverAbstract(ABC):
    @abstractclassmethod
    def __init__(self, text_corpus : list[str]) -> None:
        # Always use initialization with these arguments
        pass

    @abstractclassmethod
    def __str__(self) -> str:
        # Mention model name here
        pass

    @abstractclassmethod
    def retrieve(self) -> list[str]:
        # Return chunks here
        pass

if info_config['method'] == 'instructor':
    class Retriever(RetrieverAbstract):
        def __init__(self, text_corpus : list[str]):
            self.text_corpus = text_corpus
            self.corpus_embeddings = None
            self.model = INSTRUCTOR(huggingface_model)
            self.corpus_instruction = "Represent the document for retrieval:"
            self.query_instruction = 'Represent the question for retrieving supporting documents: '
            self.corpus_embedder()

        def __str__(self):
            return "Instructor-Large"

        def query_embedder(self,query : str):
            return self.model.encode(
                [[self.query_instruction, query]],
                show_progress_bar=False,
                batch_size=batch_size,
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
        
        def corpus_embedder(self):
            if(self.corpus_embeddings == None):
                self.corpus_embeddings = self.model.encode(
                    [[self.corpus_instruction, text] for text in self.text_corpus],
                    show_progress_bar=False,
                    batch_size=batch_size,
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                )

        def retrieve(self, query : str, top_k : int = 5) -> list[str]:
            query_embeddings = self.query_embedder(query)

            similarities = cosine_similarity(query_embeddings,self.corpus_embeddings)

            # https://stackoverflow.com/a/13070505 (Get indices of the top N values of a list)
            top_results = sorted(range(len(similarities[0])), key=lambda i: similarities[0][i], reverse=True)[:top_k]

            return [self.text_corpus[i] for i in top_results], similarities[0]

elif info_config['method'] == 'openai':
    class Retriever(RetrieverAbstract):
        def __init__(self, text_corpus : list[str]):
            source_chunks = list()
            counter = 0
            for chunk in text_corpus:
                new_metadata = {'source': str(counter)}
                source_chunks.append(Document(page_content=chunk, metadata=new_metadata))
                counter += 1
            
            self.search_index = FAISS.from_documents(source_chunks, OpenAIEmbeddings())

        def __str__(self):
            return "Open AI embeddings based FAISS"

        def rephrased_question(self, user_query):
            template = """
            Write the same question as user input and make it more descriptive without adding new information and without making the facts incorrect.

            User: {question}
            Rephrased User input:"""
            prompt = PromptTemplate(template=template, input_variables=["question"])
            llm_chain = LLMChain(prompt=prompt, llm=OpenAI(temperature=0), verbose=False)
            response = llm_chain.predict(question=user_query)
            return response.strip()

        def retrieve(self, query, top_k=5):
            paraphrased_query = self.rephrased_question(query)
            documents_with_scores = self.search_index.similarity_search_with_score(paraphrased_query, k=top_k)
            result = [x[0].page_content for x in documents_with_scores]
            scores = [x[1] for x in documents_with_scores]
            return result, scores