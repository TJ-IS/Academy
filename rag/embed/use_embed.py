from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())


import argparse
import os
import pickle
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


def main(args):
    pass
    


def dev(args):
    embeddings = OpenAIEmbeddings(model='BAAI/bge-m3')
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=args.chroma_dir,
        collection_name='academy',
    )

    query = 'What is the loss aversion?'
    docs = vectorstore.similarity_search(query, k=3)
    print(docs)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--chroma_dir', type=str, default='./export/chroma', help='The path to the chroma directory')
    args = args.parse_args()
    
    os.makedirs(args.chroma_dir, exist_ok=True)
    dev(args)