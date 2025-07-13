from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())


import argparse
import os
import pickle
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


def get_docs_by_query(query, vectorstore, k=10):
    docs = vectorstore.similarity_search(query, k=k)
    return docs


def main(args):
    embeddings = OpenAIEmbeddings(model=os.environ['EMBEDDING_MODEL'], dimensions=1024)
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=args.chroma_dir,
        collection_name='ais_basket',
    )
    
    query = 'The community building goal is achieved through the creative combinations of membership'
    results = vectorstore.similarity_search_with_score(query=query, k=3)
    for res, score in results:
        print(f"* {res.page_content}")
        print(f"[{res.metadata}]")
        print(f"Score: {score}")
        print('-' * 100)


def dev(args):
    embeddings = OpenAIEmbeddings(model=os.environ['EMBEDDING_MODEL'], dimensions=1024)
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=args.chroma_dir,
        collection_name='test',
    )

    query = 'dual-system theory'
    # docs = get_docs_by_query(query, vectorstore)
    docs = vectorstore.similarity_search(query='hello_world', k=5, filter={"section": "Introduction"})

    for doc in docs:
        print(doc.metadata)
        print(doc.page_content)
        print('-' * 100)
        print()


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--chroma_dir', type=str, default='./export/chroma', help='The path to the chroma directory')
    args.add_argument('--dev', action='store_true', help='Use dev mode')
    args = args.parse_args()
    
    os.makedirs(args.chroma_dir, exist_ok=True)
    if args.dev:
        dev(args)
    else:
        main(args)