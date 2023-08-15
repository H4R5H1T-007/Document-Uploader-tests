import textwrap
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import math
from scipy.signal import argrelextrema
import numpy as np
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

from abc import ABC, abstractclassmethod

with open("../text_segmentation_model.json") as f:
    model_config = json.load(f)

max_chunk_length = model_config['max_chunk_length']

class SplitterAbstract(ABC):
    @abstractclassmethod
    def __init__(self, doc : str, max_chunk_length : int) -> None:
        # Always use initialization with these arguments
        pass

    @abstractclassmethod
    def __str__(self) -> str:
        # Mention model name here
        pass

    @abstractclassmethod
    def predict(self) -> list[str]:
        # Return chunks here
        pass

    @staticmethod
    def text_to_sentences(text, max_chunk_length = max_chunk_length):
        sentencesInDoc = list()
        sentences = sent_tokenize(text)
        for sentence in sentences:
            currSentenceSplits = textwrap.wrap(sentence.strip(), max_chunk_length)
            sentencesInDoc.extend(currSentenceSplits)
        return sentencesInDoc

if model_config['model'] == 'entropy':
    class Splitter(SplitterAbstract):
        def __init__(self, doc, max_chunk_length = max_chunk_length):
            self.doc = doc
            self.sentencesInDoc = []
            self.max_chunk_length = max_chunk_length
            self.embedding = None
            self.docToSentences()
        
        def __str__(self):
            return 'entropy'

        def docToSentences(self):
            sentences = sent_tokenize(self.doc)
            for sentence in sentences:
                currSentenceSplits = textwrap.wrap(sentence.strip(), self.max_chunk_length)
                self.sentencesInDoc.extend(currSentenceSplits)

        def getChunksConsideringWords(self):
        ### splitting on words
            return textwrap.wrap(self.doc, self.max_chunk_length)

        def getChunksConsideringSentences(self):
        ## chunk having one sentence only. Keep on adding sentences using docToSentences until you reach maximum words threshold
            chunks = []
            curr_sentence = ''
            for sentence in self.sentencesInDoc:
                if len(curr_sentence + sentence) > self.max_chunk_length:
                    chunks.append(curr_sentence)
                    curr_sentence = sentence
                else:
                    curr_sentence = curr_sentence + ' ' + sentence
            chunks.append(curr_sentence)
            return chunks

        # taken from https://medium.com/@npolovinkin/how-to-chunk-text-into-paragraphs-using-python-8ae66be38ea6
        def predict(self,numberOfSentencesToCalculatedWeightedSum=10):
            encoding_model = SentenceTransformer(model_config['huggingface_model'])

            if self.embedding is None:
                print('Document Encoding Process :-')
                self.embedding = encoding_model.encode(
                        self.sentencesInDoc,
                        show_progress_bar=True,
                        batch_size=model_config['batch_size'],
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    )

            similarities = cosine_similarity(self.embedding)
            activated_similarities = self.activate_similarities(similarities, p_size=numberOfSentencesToCalculatedWeightedSum)
            minimas = argrelextrema(activated_similarities, np.less, order=2)[0].reshape(-1)
            return self.getChunksWithMinimas(minimas)

        def getChunksWithMinimas(self,minimas):
            chunks = []
            chunk_start = 0
            for minima in minimas:
                mergedSentence = ' '.join(self.sentencesInDoc[chunk_start:minima])
                chunk_start = minima
                currSentenceSplits = textwrap.wrap(mergedSentence, self.max_chunk_length)
                chunks.extend(currSentenceSplits)
            if(chunk_start != len(self.sentencesInDoc) - 1):
                mergedSentence = ' '.join(self.sentencesInDoc[chunk_start:])
                currSentenceSplits = textwrap.wrap(mergedSentence, self.max_chunk_length)
                chunks.extend(currSentenceSplits)
            return chunks

        def getBulletPointsFromText(self,text):
            return [s.strip('-') for s in text.split('\n') if s.strip()[0]=='-']
        def rev_sigmoid(self,x:float)->float:
            return (1 / (1 + math.exp(0.5*x)))

        def activate_similarities(self,similarities:np.array, p_size=10)->np.array:
            """ Function returns list of weighted sums of activated sentence similarities
            Args:
                similarities (numpy array): it should square matrix where each sentence corresponds to another with cosine similarity
                p_size (int): number of sentences are used to calculate weighted sum
            Returns:
                list: list of weighted sums
            """
            # To create weights for sigmoid function we first have to create space. P_size will determine number of sentences used and the size of weights vector.
            x = np.linspace(-10,10,p_size)
            # Then we need to apply activation function to the created space
            y = np.vectorize(self.rev_sigmoid)
            # Because we only apply activation to p_size number of sentences we have to add zeros to neglect the effect of every additional sentence and to match the length ofvector we will multiply
            activation_weights = np.pad(y(x),(0,similarities.shape[0]-p_size))
            ### 1. Take each diagonal to the right of the main diagonal
            diagonals = [similarities.diagonal(each) for each in range(0,similarities.shape[0])]
            ### 2. Pad each diagonal by zeros at the end. Because each diagonal is different length we should pad it with zeros at the end
            diagonals = [np.pad(each, (0,similarities.shape[0]-len(each))) for each in diagonals]
            ### 3. Stack those diagonals into new matrix
            diagonals = np.stack(diagonals)
            ### 4. Apply activation weights to each row. Multiply similarities with our activation.
            diagonals = diagonals * activation_weights.reshape(-1,1)
            ### 5. Calculate the weighted sum of activated similarities
            activated_similarities = np.sum(diagonals, axis=0)
            return activated_similarities

elif model_config['model'] == 'langchain':
    class Splitter(SplitterAbstract):
        def __init__(self, doc, max_chunk_length = max_chunk_length, chunk_overlap = model_config['chunk_overlap']):
            self.doc = [doc]
            self.max_chunk_length = max_chunk_length
            self.chunk_overlap = chunk_overlap

        def __str__(self):
            return 'langchain'

        def predict(self):
            source_chunks = list()
            splitter = RecursiveCharacterTextSplitter(chunk_size=self.max_chunk_length, chunk_overlap=self.chunk_overlap)
            for source in self.doc:
                for chunk in splitter.split_text(source):
                    source_chunks.append(chunk)

            return source_chunks

if __name__ == '__main__':
    splitter = Splitter('This is a test.', 128)
    print(splitter.predict())