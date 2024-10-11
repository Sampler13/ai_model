import os
import torch
import numpy as np
from typing import List, Tuple, Dict
from transformers import T5ForConditionalGeneration, GenerationConfig, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
import chromadb
from collections import deque

class MLModels:
    def __init__(self):
        self.vector_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.ru_llm_tokenizer = GPT2Tokenizer.from_pretrained('bond005/FRED-T5-large-instruct-v0.1')
        self.ru_llm_model = T5ForConditionalGeneration.from_pretrained('bond005/FRED-T5-large-instruct-v0.1')
        self.ru_llm_config = GenerationConfig.from_pretrained('bond005/FRED-T5-large-instruct-v0.1')
        if torch.cuda.is_available():
            self.ru_llm_model = self.ru_llm_model.cuda()

class VectorDatabase:
    def __init__(self, vector_model):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(name="documents")
        self.vector_model = vector_model
        self.doc_id_map = {} 

    def populate(self, contexts: List[Tuple[str, str]]):
        for doc_id, context in contexts:
            vector = self.vector_model.encode(context)
            self.collection.add(
                documents=[context],
                embeddings=[vector.tolist()],
                ids=[doc_id]
            )
    
    def query_best_matches(self, full_context_vector, top_n: int = 3):
        results = self.collection.query(
            query_embeddings=[full_context_vector],
            n_results=top_n
        )
        matches = [(doc_id, doc) for doc_id, doc in zip(results['ids'], results['documents'])] 
        return matches 

def load_documents_from_folder(folder_path: str) -> List[Tuple[str, str]]:
    return [
        (filename, open(os.path.join(folder_path, filename), 'r', encoding='utf-8').read().strip())
        for filename in os.listdir(folder_path) if filename.endswith('.txt')
    ]

def generate_answer(prompt: str, question: str, history: List[str], 
                    tokenizer: GPT2Tokenizer, config: GenerationConfig, 
                    model: T5ForConditionalGeneration, vector_db: VectorDatabase) -> str:
    full_context = " ".join(history + [question])
    question_vector = vector_db.vector_model.encode(full_context).tolist()

    best_matches = vector_db.query_best_matches(question_vector)  
    context_info = "\n".join([f"В документе {doc_id} содержится: {doc}" for doc_id, doc in best_matches])

    input_text = context_info + "\n" + question
    
    x = tokenizer(input_text, return_tensors='pt', padding=True).to(model.device)
    out = model.generate(**x, max_length=500, num_beams=5, temperature=0.7, early_stopping=True)
    
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()

def main():
    models = MLModels()
    vector_db = VectorDatabase(models.vector_model)

    history_size = 50
    history = deque(maxlen=history_size)

    while True:
        contexts = load_documents_from_folder('docs')
        vector_db.populate(contexts)

        question = input("Question: ")
        prompt = 'Используя этот контекст, ответь пожалуйста на мой вопрос. Постарайся дать максимально развернутый ответ, подходящий по смыслу, минимум 150 символов:'

        output = generate_answer(prompt, question, list(history),
                                models.ru_llm_tokenizer, models.ru_llm_config, 
                                models.ru_llm_model, vector_db)

        history.append(question)
        
        print(f'Вопрос: {question}\n')
        print(f'Ответ: {output}\n')

if __name__ == "__main__":
    main()
