from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

import asyncio
import argparse
import os
import random
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from tqdm.asyncio import tqdm

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from rag.embed.md_loader import get_paper_docs


async def handle_one_paper(paper_id, vectorstore, semaphore):
    with semaphore:
        paper_docs = get_paper_docs(paper_id, args)
        vectorstore.add_documents(paper_docs)
        
        # TODO: 检测是否成功添加到 vectorstore


async def main(args):
    embeddings = OpenAIEmbeddings(model=os.environ['EMBEDDING_MODEL'])
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=args.chroma_dir,
        collection_name='academy',
    )

    # 获取所有论文文档文件
    paper_ids = [str(i) for i in os.listdir(args.papers_mineru_dir)]

    # TODO: 使用一个指数分布分配每个 task 的 delay_time
    # TODO: 使用 tqdm 实时更新任务进度
    semaphore = asyncio.Semaphore(10)
    tasks = [handle_one_paper(paper_id, vectorstore, semaphore) for paper_id in paper_ids]


async def dev(args):
    embeddings = OpenAIEmbeddings(model=os.environ['EMBEDDING_MODEL'])
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=args.chroma_dir,
        collection_name='test',
    )

    paper_id = 1

    paper_docs = get_paper_docs(paper_id, args)
    print(paper_docs)

    vectorstore.add_documents(paper_docs)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--papers_mineru_dir', type=str, default='./export/papers_mineru', help='The path to the papers mineru directory')
    args.add_argument('--chroma_dir', type=str, default='./export/chroma', help='The path to the chroma directory')
    args.add_argument('--skip_existing', action='store_true', help='Skip papers that already exist in the vectorstore')
    args.add_argument('--dev', action='store_true', help='Use dev mode')
    args = args.parse_args()
    
    if not os.path.exists(args.papers_mineru_dir):
        print(f'[ERROR] Papers mineru directory {args.papers_mineru_dir} does not exist')
        exit(1)
    
    os.makedirs(args.chroma_dir, exist_ok=True)
    if args.dev:
        asyncio.run(dev(args))
    else:
        asyncio.run(main(args))