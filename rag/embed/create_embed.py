import shutil
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

import asyncio
import argparse
import os
import pickle
import random
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from tqdm.asyncio import tqdm


async def handle_one_paper(paper_doc, vectorstore, args):
    try:
        # 从文件名获取paper_id
        paper_id = paper_doc.replace('.pkl', '')
        
        # 添加随机延时，避免网络拥堵
        if args.delay_range:
            delay = random.uniform(args.delay_range[0], args.delay_range[1])
            await asyncio.sleep(delay)
        
        # 如果启用了跳过已存在数据的选项
        if args.skip_existing:
            # 检查是否已经存在该论文的文档
            existing_docs = vectorstore.get(where={"paper_id": int(paper_id)})
            if existing_docs['ids']:
                return f'Paper {paper_id} (已存在)'
        
        # 加载文档并添加到向量存储
        docs = pickle.load(open(os.path.join(args.papers_docs_dir, paper_doc), 'rb'))
        vectorstore.add_documents(docs)

        # 删除pkl文件
        pkl_file_path = os.path.join(args.papers_docs_dir, paper_doc)
        try:
            os.remove(pkl_file_path)
            # print(f'[INFO] Deleted {paper_doc}')
        except Exception as e:
            # print(f'[WARNING] Failed to delete {paper_doc}: {e}')
            pass
        return f'Paper {paper_id} (已添加)'
    except Exception as e:
        print(f'[ERROR] Failed to process paper {paper_doc}: {e}')
        return f'Paper {paper_id} (失败)'


async def main(args):
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=args.chroma_dir,
        collection_name='academy',
    )

    # 获取所有论文文档文件
    paper_docs = os.listdir(args.papers_docs_dir)
    
    # 使用信号量限制并发数
    semaphore = asyncio.Semaphore(10)
    
    async def process_with_semaphore(paper_doc):
        async with semaphore:
            return await handle_one_paper(paper_doc, vectorstore, args)
    
    # 创建任务列表，使用进度条显示
    tasks = [process_with_semaphore(paper_doc) for paper_doc in paper_docs]
    
    # 使用tqdm显示进度
    results = []
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc='Processing papers'):
        result = await f
        results.append(result)
    
    # 统计结果
    added_count = len([r for r in results if '(已添加)' in r])
    skipped_count = len([r for r in results if '(已存在)' in r])
    print(f'\n处理完成: 新增 {added_count} 篇论文, 跳过 {skipped_count} 篇已存在的论文')


async def dev(args):
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=args.chroma_dir,
        collection_name='test',
    )

    paper_docs = os.listdir(args.papers_docs_dir)[:10]

    semaphore = asyncio.Semaphore(2)

    async def process_with_semaphore(paper_doc):
        async with semaphore:
            return await handle_one_paper(paper_doc, vectorstore, args)

    tasks = [process_with_semaphore(paper_doc) for paper_doc in paper_docs]
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc='Processing papers'):
        await f


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--papers_docs_dir', type=str, default='./export/papers_docs', help='The path to the papers docs directory')
    args.add_argument('--chroma_dir', type=str, default='./export/chroma', help='The path to the chroma directory')
    args.add_argument('--skip_existing', action='store_true', help='Skip papers that already exist in the vectorstore')
    args.add_argument('--delay_range', type=float, nargs=2, metavar=('MIN', 'MAX'), default=[0.5, 2.0],
                     help='Random delay range in seconds between requests (e.g., 0.5 2.0)')
    args.add_argument('--dev', action='store_true', help='Use dev mode')
    args = args.parse_args()
    
    if not os.path.exists(args.papers_docs_dir):
        print(f'[ERROR] Papers docs directory {args.papers_docs_dir} does not exist')
        exit(1)

    os.makedirs(args.chroma_dir, exist_ok=True)
    if args.dev:
        asyncio.run(dev(args))
    else:
        asyncio.run(main(args))