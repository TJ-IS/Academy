from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

import asyncio
import argparse
import os
import random
import numpy as np
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from tqdm.asyncio import tqdm

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from rag.embed.md_loader import get_paper_docs


async def handle_one_paper(paper_id, vectorstore, semaphore, args, delay_time=0):
    """Process the embedding task of a single paper"""
    async with semaphore:
        if await is_paper_exists(paper_id, vectorstore):
            return {"paper_id": paper_id, "status": "skipped", "reason": "already_exists", "count": 0}

        try:
            await asyncio.sleep(delay_time)
            
            # Get paper documents
            paper_docs = get_paper_docs(paper_id, args)
            
            if not paper_docs:
                return {"paper_id": paper_id, "status": "skipped", "reason": "no_docs", "count": 0}
            
            # Add documents to vectorstore
            vectorstore.add_documents(paper_docs)
            
            if await is_paper_exists(paper_id, vectorstore):
                return {"paper_id": paper_id, "status": "success", "reason": "added", "count": len(paper_docs)}
            else:
                return {"paper_id": paper_id, "status": "failed", "reason": "verification_failed", "count": 0}
                
        except Exception as e:
            print(f"[ERROR] Error processing paper {paper_id}: {e}")
            return {"paper_id": paper_id, "status": "error", "reason": str(e), "count": 0}


async def process_batch(batch_papers, vectorstore, semaphore, args, batch_num):
    """Process a batch of papers"""
    batch_size = len(batch_papers)
    # Generate exponential distribution delay times for the current batch
    delay_times = np.random.exponential(scale=10.0, size=batch_size)
    
    print(f"[INFO] Starting to process batch {batch_num}, containing {batch_size} papers")
    
    # Create tasks for the current batch
    tasks = [
        handle_one_paper(paper_id, vectorstore, semaphore, args, delay_time) 
        for paper_id, delay_time in zip(batch_papers, delay_times)
    ]
    
    # Process all tasks in the current batch
    batch_results = []
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Batch {batch_num}"):
        result = await task
        batch_results.append(result)
    
    return batch_results


async def is_paper_exists(paper_id, vectorstore):
    """Check if the paper already exists in the vectorstore"""
    docs = vectorstore.similarity_search(query=f"hello_world", k=1, filter={"paper_id": paper_id})
    return len(docs) > 0


async def main(args):
    """Main function to process all papers' embedding tasks in batch"""
    embeddings = OpenAIEmbeddings(model=os.environ['EMBEDDING_MODEL'], dimensions=1024)
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=args.chroma_dir,
        collection_name='ais_basket',
    )

    # Get all paper ids
    paper_ids = [str(i) for i in os.listdir(args.papers_mineru_dir) if os.path.isdir(os.path.join(args.papers_mineru_dir, i))]
    print(f"[INFO] Found {len(paper_ids)} paper folders")

    # Split papers into batches
    batch_size = args.batch_size
    batches = []
    for batch_idx in range(0, len(paper_ids), batch_size):
        batch_paper_ids = paper_ids[batch_idx:min(batch_idx + batch_size, len(paper_ids))]
        batches.append(batch_paper_ids)
    
    print(f"[INFO] Split {len(paper_ids)} papers into {len(batches)} batches, each batch contains at most {batch_size} papers")
    
    # Create semaphore to control concurrency
    semaphore = asyncio.Semaphore(10)
    
    # Process all batches
    all_results = []
    for batch_num, batch_papers in enumerate(batches, 1):
        batch_results = await process_batch(batch_papers, vectorstore, semaphore, args, batch_num)
        all_results.extend(batch_results)
        
        # Interval between batches
        if batch_num < len(batches):
            print(f"[INFO] Batch {batch_num} completed, waiting {args.batch_interval} seconds before processing next batch...")
            await asyncio.sleep(args.batch_interval)
    
    # Statistics
    success_count = sum(1 for r in all_results if r["status"] == "success")
    failed_count = sum(1 for r in all_results if r["status"] == "failed")
    error_count = sum(1 for r in all_results if r["status"] == "error")
    skipped_count = sum(1 for r in all_results if r["status"] == "skipped")
    total_docs = sum(r["count"] for r in all_results if r["status"] == "success")
    
    print(f"[INFO] Task completion statistics:")
    print(f"[INFO] Success: {success_count}")
    print(f"[INFO] Failed: {failed_count}")
    print(f"[INFO] Error: {error_count}")
    print(f"[INFO] Skipped: {skipped_count}")
    print(f"[INFO] Total processed documents: {total_docs}")
    
    # Output error details
    if error_count > 0:
        print(f"[ERROR] Error details:")
        for result in all_results:
            if result["status"] == "error":
                print(f"  Paper {result['paper_id']}: {result['reason']}")


async def dev(args):
    """Development mode: test the processing of a single paper"""
    embeddings = OpenAIEmbeddings(model=os.environ['EMBEDDING_MODEL'], dimensions=1024)
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=args.chroma_dir,
        collection_name='test',
    )

    paper_id = "1"  # Use string format to keep consistent

    if await is_paper_exists(paper_id, vectorstore):
        print(f"[INFO] Paper {paper_id} already exists")
    else:
        paper_docs = get_paper_docs(paper_id, args)
        print(f"[INFO] Adding paper {paper_id}")
        vectorstore.add_documents(paper_docs)
        if await is_paper_exists(paper_id, vectorstore):
            print(f"[INFO] Paper {paper_id} added")
        else:
            print(f"[ERROR] Paper {paper_id} failed to add")


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--papers_mineru_dir', type=str, default='./export/papers_mineru', help='The path to the papers mineru directory')
    args.add_argument('--chroma_dir', type=str, default='./export/chroma', help='The path to the chroma directory')
    args.add_argument('--db_path', type=str, default='./export/db/academy.db', help='The path to the database file')
    args.add_argument('--dev', action='store_true', help='Use dev mode')
    args.add_argument('--batch_size', type=int, default=100, help='Number of papers to process in each batch')
    args.add_argument('--batch_interval', type=float, default=10.0, help='Interval (in seconds) between batches')
    args = args.parse_args()
    
    if not os.path.exists(args.papers_mineru_dir):
        print(f'[ERROR] Papers mineru directory {args.papers_mineru_dir} does not exist')
        exit(1)
    
    os.makedirs(args.chroma_dir, exist_ok=True)
    
    if args.dev:
        asyncio.run(dev(args))
    else:
        asyncio.run(main(args))