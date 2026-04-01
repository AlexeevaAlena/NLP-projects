import os
import pickle
import re
import dotenv
from dotenv import load_dotenv
import langchain_text_splitters
from huggingface_hub import InferenceClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from loguru import logger
import json
# from tqdm import tqdm
from threading import Lock
from collections import Counter
import pandas as pd
# import re
import sys

import numpy as np
import faiss
# import getpass
import langchain_community
import langgraph
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph
import requests
# import os.path
# import langchain_text_splitters.html
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
# import operator
from typing_extensions import TypedDict
from typing import List, Annotated
from langgraph.graph import END
from openai import OpenAI
from transformers import pipeline
from typing import TypedDict, Optional
import os
import sys
import torch
import pandas as pd
from loguru import logger
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from sentence_transformers import SentenceTransformer
import sqlite3
import pandas as pd
import re
from transformers import pipeline, GenerationConfig

# ПРОМПТЫ:
# -------------------------------------------


rag_prompt = """Ты - опытный сотрудник приемной комиссии университета, помогающий абитуриентам и их родителям разобраться в вопросах поступления, обучения и студенческой жизни. Ты общаешься доброжелательно и профессионально.

Твоя задача - давать четкие, структурированные ответы, которые помогут абитуриенту принять верное решение. Избегай излишней бюрократизации, но сохраняй официально-деловой тон там, где это уместно.

Вот информация, которую ты можешь использовать для ответа на вопрос:
{context} 

Внимательно изучи предоставленный контекст. В ответе не нужно ссылаться на то, что ты используешь какую-то базу знаний - просто отвечай как специалист.

Теперь ответь на вопрос абитуриента:

{question}

Правила подготовки ответа:
1. Будь максимально точен - опирайся только на факты из контекста
2. Структурируй информацию: используй списки, пункты, выделяй важные детали
3. Если в контексте есть цифры (баллы, места, стоимость) - обязательно их приведи
4. Отвечай ясным русским языком, без сложных канцеляризмов

Структура ответа (если применимо):
- Начни с приветствия: "Здравствуйте! Рад помочь с вопросом о поступлении."
- Дай прямой ответ на вопрос
- Заверши четко: предложи следующий шаг или уточни, что можно обратиться за дополнительной информацией

Ответ должен состоять из 3-4 предложений НЕ БОЛЬШЕ!

Пример:
Вопрос: "Какие экзамены нужно сдавать на программиста?"
Ответ: "Здравствуйте! Для поступления на направление 'Программная инженерия' необходимо сдать ЕГЭ по следующим предметам: русский язык, математика (профиль) и информатика. Это три обязательных экзамена. Минимальный проходной балл в прошлом году составил 210. Прием документов начинается 20 июня и заканчивается 25 июля. Если у вас есть индивидуальные достижения (победы в олимпиадах, золотая медаль), вы можете получить до 10 дополнительных баллов. Если остались вопросы, обращайтесь в приемную комиссию по телефону или приходите на день открытых дверей 15 апреля."

Твой ответ:"""

hallucination_grader_prompt = """ФАКТЫ: \n\n {documents} \n\n ОТВЕТ СТУДЕНТА: {generation}. 
Ответь в формате JSON (не текстом, а именно json-файлом) с помощью двух ключей, первый это  binary_score - это оценка "yes" или "no", чтобы указать, основан ли ОТВЕТ СТУДЕНТА на ФАКТАХ. 
Второй ключ это explanation - пояснение, которое содержит объяснение поставленного  binary_score.
"""

hallucination_grader_instructions = """

Ты являешься строгим преподавателем, оценивающим ответ студента на основе предоставленного фактического материала.
На вход ты получишь два артефакта:
- ФАКТЫ: ключевая фактическая информация, извлеченная из документов.
- ОТВЕТ СТУДЕНТА: сгенерированный ответ на вопрос пользователя.

Твоя цель - определить, подтверждается ли ОТВЕТ СТУДЕНТА ФАКТАМИ. То есть, отражает ли ОТВЕТ СТУДЕНТА информацию из ФАКТОВ. И была ли сохранена логика.

Используй следующие критерии:

1. Ответ не обязательно должен в точности повторять ФАКТЫ, но он должен отражать то же значение или не противоречить им.
2. Допустимо, чтобы ОТВЕТ СТУДЕНТА включал легкое перефразирование, обобщение или разумные выводы, если они
основаны на предоставленном содержании.
3. Не помечай ответ как неправильный, если только он явно не содержит выдуманной, неподтвержденной или противоречивой информации

"""

answer_grader_prompt = """ВОПРОС: \n\n {question} \n\n ОТВЕТ УЧЕНИКА: {generation}. 

Верни JSON с двумя ключами: binary_score это 'yes' или 'no'  - скор, который показывает, подходит ли ОТВЕТ УЧЕНИКА под критерии. 
И explanation, объяснение, которое поясняет выставленный binary_score."""

answer_grader_instructions = """

    Ты - преподаватель, оценивающий результаты теста. Тебе будет показан ВОПРОС, заданный УЧЕНИКУ, и ОТВЕТ УЧЕНИКА. 
    Ниже приведены критерии выставления оценки, которым тебе необходимо следовать:

    (1) ОТВЕТ УЧЕНИКА помогает ответить на поставленный ВОПРОС. 
    (2) УЧЕНИК не отвечает на острополитический, аморальный ВОПРОС. УЧЕНИКУ категорически нельзя отвечать на ВОПРОСЫ, в которых затрагивается
    расовая принадлежность с аморальной точки зрения, черный юмор, непристойные темы, остро политические вопросы

    Правила выставления оценки УЧЕНИКУ:

    Оценка "yes" означает, что ответ учащегося соответствует всем критериям. Это наивысший балл. 

    Учащийся может получить оценку "yes", если ответ содержит дополнительную информацию, которая явно не указана в вопросе. Но информация не должна
    перечить общим жизненным закономерностям. 

    Оценка "no" означает, что ответ учащегося не соответствует всем критериям. Это максимально низкий балл, который ты можешь выставить.

    Объясни свои рассуждения поэтапно, чтобы убедиться в правильности своих рассуждений и выводов. 

    Избегай простого указания правильного ответа в начале.

    """

HF_TOKEN = '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'
# ================= LOGGING =================
LOG_PATH = "rag_logs.log"
logger.remove()
logger.add(sys.stderr, level="DEBUG")
logger.add(LOG_PATH, rotation="10 MB", retention="30 days")

load_dotenv()

# ================= EMBEDDING =================
client = InferenceClient(token=HF_TOKEN)
model = SentenceTransformer("intfloat/multilingual-e5-base")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

class EmbeddingWrapper:
    def embed_documents(self, texts):
        texts = [f"passage: {t}" for t in texts]
        return model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, query):
        query = f"query: {query}"
        return model.encode(query, normalize_embeddings=True).tolist()

    def __call__(self, text):
        # ЭТО КЛЮЧЕВОЕ
        return self.embed_query(text)

# ================= LOADERS =================
def load_excel(file_path: str):
    df = pd.read_excel(file_path)
    documents = []
    for _, row in df.iterrows():
        text = " ".join([str(x) for x in row if pd.notna(x)])

        documents.append(
            Document(
                page_content=text,
                metadata={"source": file_path}
            )
        )

    return documents

def load_excel_faq(file_path: str):
    df = pd.read_excel(file_path)

    documents = []
    for _, row in df.iterrows():
        text = str(row["Question"])  # ← ТОЛЬКО вопрос

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": file_path,
                    "answer": row["Answer"],
                    "type": row["Question type"]
                }
            )
        )
    return documents

def load_text_files(file_paths: list) -> list:
    all_words = set()  # используем set для автоматического удаления дубликатов

    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip().lower()
                    if word:  # пропускаем пустые строки
                        all_words.add(word)
        except FileNotFoundError:
            print(f"Файл {file_path} не найден, пропускаем...")

    return list(all_words)


# ================= SPLIT =================
def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


# ================= INDEX BUILDER =================
def build_faiss_index(file_path: str, index_path: str):
    embedding_wrapper = EmbeddingWrapper()

    # загрузка данных
    if file_path.endswith(".xlsx"):
        if file_path=='data/Database.xlsx': #если faq то не разбиваем на чанки
            documents = load_excel_faq(file_path)
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
        else:
            documents = load_excel(file_path)
            split_docs = split_documents(documents)

            texts = [d.page_content for d in split_docs]
            metadatas = [d.metadata for d in split_docs]
    else:
        raise ValueError(f"Неизвестный формат файла: {file_path}")

    logger.debug(f"{file_path}: загружено документов {len(documents)}")

    logger.debug(f"{file_path}: чанков {len(texts)}")

    # создание FAISS
    db = FAISS.from_texts(
        texts=texts,
        embedding=embedding_wrapper,
        metadatas=metadatas
    )

    # сохранение
    os.makedirs(index_path, exist_ok=True)
    db.save_local(index_path)

    logger.debug(f"Индекс сохранён: {index_path}")
    return db


# ================= BUILD ALL INDEXES =================
def build_all_indexes():
    prepared_data = {'stop_words':load_text_files(["data/ru_abusive_words.txt","data/ru_curse_words.txt"])}
    datasets = {
        "data/Database.xlsx": "faiss_database",
        "data/Database-2.xlsx": "faiss_database_2",
    }

    for file_name, index_name in datasets.items():
        if not os.path.exists(file_name):
            logger.warning(f"Файл не найден: {file_name}")
            continue
        prepared_data[index_name] = build_faiss_index(file_name, index_name)
    return prepared_data

# ================ Классификатор для SQL OR RAG =================
class ProgramClassifier:
    def __init__(self, model_path="models/program_classifier.pkl"):
        self.model_path = model_path
        self.vectorizer = None
        self.classifier = None
        self.is_trained = False

    def train(self, dff=None, text_column='question', label_column='label',
              test_size=0.2, random_state=42, save_model=True):
        db2 = pd.read_excel('data/Database-2.xlsx')
        ap = pd.read_excel('data/all_program.xlsx')

        rag_q = db2.iloc[:, 0].copy()
        sql_q = ap.iloc[:, 0].copy()

        # Создаем DataFrame из rag_q и сразу добавляем колонку
        rag_df = pd.DataFrame(rag_q)
        rag_df = rag_df.rename(columns={rag_df.columns[0]: 'question'})
        rag_df['label'] = 'rag'

        sql_df = pd.DataFrame(sql_q)
        sql_df = sql_df.rename(columns={sql_df.columns[0]: 'question'})
        sql_df['label'] = 'sql'

        dff = pd.concat([sql_df, rag_df], ignore_index=True)

        # Подготовка данных
        texts = dff[text_column].tolist()
        labels = dff[label_column].tolist()
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )

        # Создаем классификатор
        self.classifier = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=random_state
        )

        X_vec = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X_vec, labels)
        self.is_trained = True

        # Сохраняем модель если нужно
        if save_model:
            self.save()

    def predict(self, question):
        if not self.is_trained:
            raise ValueError("Модель не обучена. Сначала вызовите train() или load()")

        questions_vec = self.vectorizer.transform(question)
        prediction = self.classifier.predict(questions_vec)
        return prediction

    def save(self, custom_path=None):
        """
        Сохранение модели на диск
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена. Сначала вызовите train()")

        save_path = custom_path or self.model_path

        # Создаем директорию если не существует
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Сохраняем модель
        with open(save_path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'classifier': self.classifier,
                'classes': self.classifier.classes_,
                'version': '1.0'
            }, f)

        print(f"✅ Модель сохранена: {save_path}")
        return save_path

    def load(self, custom_path=None):
        """
        Загрузка модели с диска
        """
        load_path = custom_path or self.model_path

        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Модель не найдена: {load_path}")

        with open(load_path, 'rb') as f:
            model_data = pickle.load(f)
            self.vectorizer = model_data['vectorizer']
            self.classifier = model_data['classifier']
            self.is_trained = True

        print(f"✅ Модель загружена: {load_path}")
        print(f"   Версия: {model_data.get('version', 'unknown')}")
        print(f"   Классы: {self.classifier.classes_}")

        return self


classifier = ProgramClassifier(model_path="models/program_classifier.pkl")
# Проверяем, существует ли уже сохраненная модель
if os.path.exists("models/program_classifier.pkl"):
    print("📦 Загрузка существующей модели...")
    classifier.load()
else:
    print("🚀 Обучение новой модели...")
    classifier.train(save_model=True)
    print("✅ Модель обучена и сохранена")

### =================== SQL PART GENERATION ====================
class SQLGenerator:
    def __init__(self):
        self.pipe = None
        self.db_path = "programs.db"

    def load_model(self):
        if self.pipe is None:
            print("🔄 Загрузка модели...")
            self.pipe = pipeline(
                "text-generation",
                model="Qwen/Qwen2.5-1.5B-Instruct",
                token=HF_TOKEN,
                device_map="auto",
                dtype=torch.float16
            )
            print("✅ Модель готова")
        return self.pipe

    def generate_sql(self, question):
        pipe = self.load_model()

        messages = [
            {"role": "system", "content": """Ты опытный SQL специалист, который исходя из вопроса пользователя выдает только SQL-запрос, помогающий пользователю.
                ПРАВИЛА:
                1. Возвращай ТОЛЬКО конкретные колонки, НЕ используй SELECT *
                2. Если вопрос о цене — верни ТОЛЬКО cost и program
                3. Если вопрос о бюджетном образовании — верни ТОЛЬКО eges_budget, budget_2025 и program
                4. Если вопрос о платном обучении — верни ТОЛЬКО eges_contract, contract_2025, cost и program
                5. Если вопрос о поступлении без уточнения платное или бюджетное образование — верни ТОЛЬКО eges_contract, contract_2025, cost,eges_budget, budget_2025 и program
                6. Если вопрос о проходном балле — верни ТОЛЬКО pass_2024 и program
                7. ВСЕГДА используй LIKE '% %' для фильтрации (никаких '=')
                8. Добавляй LIMIT 5 если вопрос о списке программ или нескольких программах сразу
                9. Используй ТОЛЬКО русские слова для фильтрации

             ПРИМЕР:
             Вопрос: сколько стоит обучение на программе анализ данных?
             Ответ: SELECT cost FROM all_programs WHERE program LIKE '%анализ данных%'

             Вопрос: какие программы есть в институте информационных технологий?
             Ответ: SELECT program FROM all_programs WHERE institute LIKE '%информационных технологий%'

             Вопрос: сколько бюджетных мест на программе искусственный интеллект?
             Ответ: SELECT budget_2025 FROM all_programs WHERE program LIKE '%искусственный интеллект%'"""},
            {"role": "user", "content": f""" У тебя есть следующая таблица: all_programs. В ней содержаться следующие колонки: program (программа обучения - все программы указаны с маленькой буквы), 
            cost (цена обучения), budget_2025(количество бюджетных мест в 2025 году), contract_2025(количество платных мест в 2025 году)
            pass_2024(Проходной балл по ЕГЭ в 2024), decs (Описание программы)), edu_form (форма обучения - очная/заочная/очно-заочная),
            eges_contract(условия для поступления на платное обучение), eges_budget (условия для постпуления на бюджетное место).
            Тебе пришел запрос от пользователя: Вопрос: {question}"""}
        ]

        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        response = pipe(prompt, max_new_tokens=80, temperature=0.1)

        full = response[0]['generated_text']
        if "<|im_start|>assistant" in full:
            sql = full.split("<|im_start|>assistant")[-1].strip()
        else:
            sql = full.split("SQL:")[-1].strip()

        sql = re.sub(r"LIKE '%([^%]+)%'", lambda m: f"LIKE '%{re.sub(r'[A-Za-z]', '', m.group(1))}%'",
                     sql) if re.search(r"[A-Za-z]", sql) else sql
        return sql

    def init_database(self, excel_path="data/all_program.xlsx"):
        dff = pd.read_excel(excel_path)
        conn = sqlite3.connect(self.db_path)
        dff.to_sql("all_programs", conn, if_exists="replace", index=False)
        conn.close()

    def query_database(self, sql_query):
        conn = sqlite3.connect(self.db_path)
        result_df = pd.read_sql_query(sql_query, conn)
        conn.close()
        return result_df

    def llm_answer(self, question, result_df):
        pipe = self.load_model()
        messages = [
            {"role": "system", "content": """ Ты помогаешь абитуриентам президенсткой академии узнать ВСЕ об интересуемых программах, тебе на вход поступает вопрос
            абитуриента
            и датафрейм с информацией, которую тебе передала другая нейронка. У тебя ОЧЕНЬ ответственная задача: тебе необходимо опираясь на данную тебе табличку сгенерировать ответ абитуриенту.
            ЗАПОМНИ: НЕ НУЖНО ПОКАЗЫВАТЬ ТАБЛИЦУ АБИТУРИЕНТУ. ТВОЯ ЗАДАЧА - ТОЛЬКО ОТВЕТИТЬ КРАСИВО И ТОЧНО НА ВОПРОС. Если вопрос про описание программы, не скупись на слова. 
            ВАЖНО:
            1. Если тебе попали численные данные ты ни в коем случае не должен менять их значение!!!! Числа менять нельзя. 
            2. Также не нужно расшифровывать аббревиатуры или сокращения - например: ИЭМИТ, ИОН 
            3. Если вопрос пользователя содержит такие слова как: Сколько, какой, как, какое число. Старайся дать краткий исчерпывающий ответ
            4. Не придумывай информацию, которой нет в предоставленных тебе данных
            ПРИМЕР ТВОЕЙ РАБОТЫ:
            Вопрос пользователя: Сколько бюджетных мест на программе анализ данных?
            Табличка содержит число 12.
            Твой ответ: На программе анализ данных всего 12 бюджетных мест."""},
            {"role": "user",
             "content": f"""  У тебя есть следующая табличка с ФАКТАМИ: {result_df}. Вопрос абитуриента: {question}"""}
        ]

        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        response = pipe(prompt, max_new_tokens=500, temperature=0.1)

        full = response[0]['generated_text']
        if "<|im_start|>assistant" in full:
            text = full.split("<|im_start|>assistant")[-1].strip()
        else:
            text = full.split("SQL:")[-1].strip()
        text = 'Я побывал в мире sql запросов и нашел для тебя ответ.' + text
        return text

    def sql_step(self, question):
        sql_query = self.generate_sql(question)
        self.init_database()
        result_df = self.query_database(sql_query)
        answer = self.llm_answer(question, result_df)
        return answer, sql_query

sql_generator = SQLGenerator()

#### ========== TOXIC ===================

toxic_classifier = pipeline(
      "text-classification",
      model="chgk13/tiny_russian_toxic_bert"
  )
def toxity_classification(query):
  results = toxic_classifier(query)
  label = results[0]['label']
  return label

# ========== Инициализация модели ==========
class LocalLLM:
    """Класс для работы с локальной моделью через Hugging Face"""

    def __init__(self):
        self.pipe = None
        self.model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    def load_model(self):
        if self.pipe is None:
            print("🔄 Загрузка модели...")
            self.pipe = pipeline(
                "text-generation",
                model=self.model_name,
                token=HF_TOKEN,  # ваш токен Hugging Face
                device_map="auto",
                dtype=torch.float16
            )
            print("✅ Модель готова")
        return self.pipe

    def generate(self, prompt_text: str, max_new_tokens: int = 500, temperature: float = 0.1) -> str:
        """Генерация текста"""
        pipe = self.load_model()

        messages = [
            {"role": "user", "content": prompt_text}
        ]

        prompt = pipe.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.95 if temperature > 0 else 1.0,
        )

        response = pipe(prompt, generation_config=generation_config)

        full = response[0]['generated_text']
        # Извлекаем ответ после промпта
        if "<|im_start|>assistant" in full:
            result = full.split("<|im_start|>assistant")[-1].strip()
        else:
            result = full.replace(prompt, "").strip()

        return result

    def generate_json(self, prompt_text: str, system_text: str = "", max_new_tokens: int = 500) -> dict:
        """Генерация JSON-ответа"""
        pipe = self.load_model()

        messages = []
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": prompt_text})

        prompt = pipe.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=0.1,  # низкая температура для JSON
            do_sample=False,  # детерминированный вывод
        )

        response = pipe(prompt, generation_config=generation_config)

        full = response[0]['generated_text']
        if "<|im_start|>assistant" in full:
            result = full.split("<|im_start|>assistant")[-1].strip()
        else:
            result = full.replace(prompt, "").strip()

        # Очистка от markdown
        result = result.replace("```json", "").replace("```", "").strip()

        try:
            return json.loads(result)
        except json.JSONDecodeError as e:
            logger.warning(f"Ошибка парсинга JSON: {result}, error: {e}")
            return {"binary_score": "no", "explanation": f"Ошибка парсинга: {result}"}


# ========== Создаем глобальный экземпляр ==========
llm = LocalLLM()
# ================ автоответ и FAQ ===========

def autoanswer(state):
    """
    Автоматический ответ на некорректные вопросы.
    """
    logger.debug("---АВТООТВЕТ НА НЕКОРРЕКТНЫЙ ВОПРОС---")
    return {"generation": "Я еще не научился отвечать на подобные вопросы ):"}

def faq(state):
    """
    Автоматический ответ на FAQ вопросы.
    """
    question=state['question']
    faq = df['faiss_database']

    results = faq.similarity_search_with_score(question, k=1)
    for doc, score in results:
        document_text = doc.metadata['answer']
    logger.debug("---FAQ НА ВОПРОС---")
    return {"generation": document_text}


def sql_node(state):
    """
    Узел для обработки SQL запросов
    Использует ваш SQLGenerator
    """
    question = state['question']

    try:
        # Используем ваш метод sql_step
        answer, sql_query = sql_generator.sql_step(question)

        # Сохраняем результаты в состояние
        state['generation'] = answer
        state['sql_query'] = sql_query

        print(f"📊 Сгенерированный SQL: {sql_query}")

    except Exception as e:
        print(f"❌ Ошибка при генерации SQL: {e}")
        state['generation'] = "Извините, произошла ошибка при обработке вашего запроса."
        state['sql_query'] = None

    return {'generation': answer}

def rag(state):
    logger.debug("---АВТООТВЕТ НА НЕКОРРЕКТНЫЙ ВОПРОС---")
    return {"generation": "Здесь будет rag"}

# ============================================
# если faiss еще не создан - раскоментить
# ============================================

# df = build_all_indexes()
# # #
# with open('df_indexes.pkl', 'wb') as f:
#        pickle.dump(df, f)
# print("✅ Объект сохранен в df_indexes.pkl")

# ============================================
# ЗАГРУЗКА если faiss создан
# ============================================

def load_saved_df():
     """Загружает сохраненный объект"""
     if os.path.exists('df_indexes.pkl'):
         with open('df_indexes.pkl', 'rb') as f:
             return pickle.load(f)
     else:
         return build_all_indexes()

 #global df
df = load_saved_df()


### ============== основной граф ===================
class GraphState(TypedDict):
    question: str
    generation: str
    route_type: Optional[str]
    sql_query: Optional[str]  # для хранения сгенерированного SQL
    sql_result: Optional[pd.DataFrame]

    autoanswer: str  # Двоичное решение об ответе (можем ли ответить в прицнипе)
    max_retries: int  # Максимальное количество повторных попыток генерации
    answers: int  # Количество сгенерированных ответов
    loop_step: int
    documents: List[str]  # Список найденных документов


## ========= ROUTER ========
def route_node(state):
    """Узел роутера - сохраняет тип"""
    question = state['question']
    stop_words = df['stop_words']
    faq = df['faiss_database']

    ### bad words
    question_clean = re.sub(r'[^\w\s]', '', question.lower())
    question_words = set(question_clean.split())
    if bool(question_words.intersection(stop_words)) == True:
        state['route_type'] = "autoanswer"
        return state

    if toxity_classification(question) == 'toxic':
        state['route_type'] = "autoanswer"
        return state

    ### faq
    results = faq.similarity_search_with_score(question, k=1)
    for doc, score in results:
        cosine_similarity = 1 - (score ** 2) / 2
        if cosine_similarity > 0.98:
            state['route_type'] = "faq"
            return state

    ### sql
    pred = classifier.predict([question])
    if pred[0] == 'sql':
        state['route_type'] = "sql"
        return state

    state['route_type'] = "rag"
    return state


def route_condition(state):
    """Условие перехода"""
    return state['route_type']

def retrieve(state):
    """
    Получение документов
    """
    logger.debug(f"Ключи состояния: {list(state.keys())}")

    question = state["question"]
    retriever = df['faiss_database_2'].as_retriever(search_kwargs={"k":3})
    documents = retriever.invoke(question)

    logger.debug(f'documents= {documents}')

    return {"documents": documents}

def reranke(state):
    logger.debug('---РАНЖИРОВАНИЕ ДОКУМЕНТОВ (локально)---')

    documents = state.get('documents', [])
    if not documents:
        logger.debug("Нет документов для реранкинга, возвращаем автоответ.")
        return {"autoanswer": "Yes"}

    text_docs = []
    for d in documents:
        if hasattr(d, "page_content"):
            text_docs.append(str(d.page_content))
        elif isinstance(d, dict) and "page_content" in d:
            text_docs.append(str(d["page_content"]))
        else:
            text_docs.append(str(d))

    question = state.get("question", "")
    logger.debug(f"Вопрос: {question[:100]}")

    pairs = [(question, doc_text) for doc_text in text_docs]
    scores = cross_encoder.predict(pairs)

    best_index = max(range(len(scores)), key=lambda i: scores[i])
    best_document = documents[best_index]
    best_score = scores[best_index]
    best_snippet = text_docs[best_index][:100]

    logger.debug(f"Лучший документ после реранкинга:")
    logger.debug(f"Score: {best_score:.4f} | Text snippet: {best_snippet}")

    return {"document": best_document}


def generate(state):
    """
    Генерация ответа на основе извлеченных документов
    """
    logger.debug("---СГЕНЕРИРОВАТЬ---")

    loop_step = state.get("loop_step", 0)
    feedback = state.get("feedback", "")

    rag_prompt_formatted = rag_prompt.format(
        context=state["documents"],
        question=state["question"] + (
            "\nТвой прошлый ответ мне не понравился. И вот почему: " + feedback if feedback else "")
    )

    generation = llm.generate(rag_prompt_formatted, max_new_tokens=500, temperature=0.1)
    logger.debug(f'generation={generation}')

    return {"generation": generation, "loop_step": loop_step + 1}

def decide_to_generate(state):
    """
    Решение о том, по какому пути продолжить генерацию ответа.

    Args:
        state (dict): Текущее состояние графа

    Returns:
        str: Двоичное решение для следующего узла вызова
    """
    logger.debug("---РЕЛЕВАНТНЫ ЛИ ДОКУМЕНТЫ?---")
    autoanswer = state.get("autoanswer", "No")
    if autoanswer == "Yes":
        logger.debug(
            "---БЫЛ ВЫБРАН АВТОМАТИЧЕСКИЙ ОТВЕТ---"
        )
        return "autoanswer"
    else:
        logger.debug("---РЕШЕНИЕ: ГЕНЕРИРОВАТЬ---")
        return "generate"


def judge_model(instructions: str, prompt_text: str) -> str:
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": prompt_text}
    ]

    response = client.chat_completion(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        messages=messages,
        max_tokens=200,
        temperature=0.1
    )
    return response.choices[0].message.content


def grade_generation_v_documents_and_question(state):
    """
    Проверка соответствия сгенерированного ответа документам и вопросу.
    """
    logger.debug("---ПРОВЕРИТЬ ГАЛЛЮЦИНАЦИИ---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    max_retries = state.get("max_retries", 3)
    loop_step = state.get("loop_step", 0)

    # Проверка галлюцинаций (генерация соответствует документам?)
    hallucination_prompt = hallucination_grader_prompt.format(
        documents=documents, generation=generation
    )

    result = llm.generate_json(
        prompt_text=hallucination_prompt,
        system_text=hallucination_grader_instructions,
        max_new_tokens=200
    )

    grade = result.get("binary_score", "no")
    explanation = result.get("explanation", "")

    logger.debug(f"---ОЦЕНКА ГАЛЛЮЦИНАЦИИ: {grade} --- ОБЪЯСНЕНИЕ: {explanation}")

    if grade.lower() == "yes":
        logger.debug("---РЕШЕНИЕ: ГЕНЕРАЦИЯ ОСНОВАНА НА ДОКУМЕНТАХ---")
        logger.debug("---Оценка: ГЕНЕРАЦИЯ против ВОПРОСА---")

        # Проверка, отвечает ли генерация на вопрос
        answer_grader_prompt_formatted = answer_grader_prompt.format(
            question=question, generation=generation
        )

        result = llm.generate_json(
            prompt_text=answer_grader_prompt_formatted,
            system_text=answer_grader_instructions,
            max_new_tokens=200
        )

        grade_answer = result.get("binary_score", "no")

        if grade_answer.lower() == "yes":
            logger.debug("---РЕШЕНИЕ: GENERATION ОБРАЩАЕТСЯ К ВОПРОСУ---")
            return "useful"
        elif loop_step <= max_retries:
            logger.debug("---РЕШЕНИЕ: GENERATION НЕ ОТВЕЧАЕТ НА ВОПРОС---")
            return "not useful"
        else:
            logger.debug("---РЕШЕНИЕ: МАКСИМАЛЬНОЕ КОЛИЧЕСТВО ПОВТОРНЫХ ПОПЫТОК ДОСТИГНУТО---")
            return "max retries"

    elif loop_step <= max_retries:
        logger.debug("---РЕШЕНИЕ: ГЕНЕРАЦИЯ НЕ ОСНОВАНА НА ДОКУМЕНТАХ, ПОВТОРИТЕ ПОПЫТКУ---")
        return "not supported"
    else:
        logger.debug("---РЕШЕНИЕ: МАКСИМАЛЬНОЕ КОЛИЧЕСТВО ПОВТОРНЫХ ПОПЫТОК ДОСТИГНУТО---")
        return "max retries"

workflow = StateGraph(GraphState)

workflow.add_node("route", route_node)  # узел
workflow.add_node("autoanswer", autoanswer)
workflow.add_node("faq", faq)
workflow.add_node("sql", sql_node)
workflow.add_node("retrieve", retrieve)
workflow.add_node("reranker", reranke)
workflow.add_node("generate", generate)


workflow.set_entry_point("route")

workflow.add_conditional_edges(
    "route",
    route_condition,  # ← условие (возвращает строку)
    {
        "autoanswer": "autoanswer",
        "faq": "faq",
        "sql": "sql", #функция для sql
        "rag": "retrieve" #ретривер
    }
)

workflow.add_edge("autoanswer", END)
workflow.add_edge("faq", END)
workflow.add_edge("sql", END)
workflow.add_edge("retrieve", 'reranker')

workflow.add_conditional_edges(
    "reranker",
    decide_to_generate,
    {
        "autoanswer": "autoanswer",
        "generate": "generate",
    },
)
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "autoanswer",
        "max retries": END,
    },
)
graph = workflow.compile()

# # Запуск workflow
# initial_state = {
#     'question': "Кто такой Марголин Андрей Маркович- Проректор?",
#     'generation': None,
#     'max_retries':2}
#
# # Выполнение
# result = graph.invoke(initial_state)

import pandas as pd

# Загружаем таблицу (пример: CSV или Excel)
test = pd.read_excel("vopros_otvet (1).xlsx")  # или pd.read_csv("testing.csv")


answers = []

for i, row in test.iterrows():
    question = row['Question']

    initial_state = {
        'question': question,
        'generation': None,
        'max_retries': 2
    }

    try:
        result = graph.invoke(initial_state)
        answer = result.get('generation', None)
    except Exception as e:
        answer = f"ERROR: {e}"

    answers.append(answer)
    print(f"[{i}] Готово")

# Записываем ответы в новый столбец
test['model_answer'] = answers

# Сохраняем обратно
test.to_excel("testing_with_answers.xlsx", index=False)



