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
    """处理单个论文的embedding任务"""
    async with semaphore:
        try:
            # 添加指数分布的延迟时间
            if delay_time > 0:
                await asyncio.sleep(delay_time)
            
            # 获取论文文档
            paper_docs = get_paper_docs(paper_id, args)
            
            if not paper_docs:
                return {"paper_id": paper_id, "status": "skipped", "reason": "no_docs", "count": 0}
            
            # 检查是否需要跳过已存在的文档
            if args.skip_existing:
                existing_docs = vectorstore.similarity_search(
                    query=f"paper_id:{paper_id}", 
                    k=5, 
                    filter={"paper_id": str(paper_id)}
                )
                if existing_docs:
                    return {"paper_id": paper_id, "status": "skipped", "reason": "already_exists", "count": len(existing_docs)}
            
            # 添加文档到vectorstore
            vectorstore.add_documents(paper_docs)
            
            # 检测是否成功添加到 vectorstore
            verification_docs = vectorstore.similarity_search(
                query=f"paper_id:{paper_id}", 
                k=5, 
                filter={"paper_id": str(paper_id)}
            )
            
            if verification_docs:
                return {"paper_id": paper_id, "status": "success", "reason": "added", "count": len(paper_docs)}
            else:
                return {"paper_id": paper_id, "status": "failed", "reason": "verification_failed", "count": 0}
                
        except Exception as e:
            print(f"Error processing paper {paper_id}: {e}")
            return {"paper_id": paper_id, "status": "error", "reason": str(e), "count": 0}


async def main(args):
    """主函数：处理所有论文的embedding任务"""
    embeddings = OpenAIEmbeddings(model=os.environ['EMBEDDING_MODEL'])
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=args.chroma_dir,
        collection_name='academy',
    )

    # 获取所有论文文档文件
    paper_ids = [str(i) for i in os.listdir(args.papers_mineru_dir) if os.path.isdir(os.path.join(args.papers_mineru_dir, i))]
    
    print(f"找到 {len(paper_ids)} 个论文文件夹")

    # 使用指数分布分配每个 task 的 delay_time
    delay_times = np.random.exponential(scale=1.0, size=len(paper_ids))
    
    # 创建信号量控制并发数
    semaphore = asyncio.Semaphore(10)
    
    # 创建任务列表
    tasks = [
        handle_one_paper(paper_id, vectorstore, semaphore, args, delay_time) 
        for paper_id, delay_time in zip(paper_ids, delay_times)
    ]
    
    # 使用 tqdm 实时更新任务进度
    print("开始处理论文embedding任务...")
    results = []
    
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="处理论文"):
        result = await task
        results.append(result)
    
    # 统计结果
    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = sum(1 for r in results if r["status"] == "failed")
    error_count = sum(1 for r in results if r["status"] == "error")
    skipped_count = sum(1 for r in results if r["status"] == "skipped")
    total_docs = sum(r["count"] for r in results if r["status"] == "success")
    
    print(f"\n任务完成统计:")
    print(f"成功: {success_count}")
    print(f"失败: {failed_count}")
    print(f"错误: {error_count}")
    print(f"跳过: {skipped_count}")
    print(f"总计处理文档数: {total_docs}")
    
    # 输出错误详情
    if error_count > 0:
        print(f"\n错误详情:")
        for result in results:
            if result["status"] == "error":
                print(f"  论文 {result['paper_id']}: {result['reason']}")


async def dev(args):
    """开发模式：测试单个论文的处理"""
    embeddings = OpenAIEmbeddings(model=os.environ['EMBEDDING_MODEL'])
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=args.chroma_dir,
        collection_name='test',
    )

    paper_id = "1"  # 使用字符串格式保持一致

    paper_docs = get_paper_docs(paper_id, args)
    print(f"获取到 {len(paper_docs)} 个文档段落")
    
    if paper_docs:
        print("文档示例:")
        for i, doc in enumerate(paper_docs[:2]):  # 只显示前两个
            print(f"  文档 {i+1}:")
            print(f"    内容长度: {len(doc.page_content)}")
            print(f"    元数据: {doc.metadata}")
            print(f"    内容预览: {doc.page_content[:100]}...")
            print()

        vectorstore.add_documents(paper_docs)
        print("文档已添加到vectorstore")
        
        # 验证添加是否成功 - 使用str类型的paper_id进行过滤
        print(f"正在验证paper_id={paper_id}的文档...")
        verification_docs = vectorstore.similarity_search(
            query=f"paper_id:{paper_id}", 
            k=5, 
            filter={"paper_id": str(paper_id)}  # 使用str类型，因为metadata中存储的是str
        )
        print(f"验证结果: 找到 {len(verification_docs)} 个相关文档")
        
        if verification_docs:
            print("找到的文档metadata:")
            for i, doc in enumerate(verification_docs):
                print(f"  文档 {i+1}: {doc.metadata}")
        else:
            # 如果没找到，尝试不使用filter进行搜索
            print("没有找到匹配的文档，尝试不使用filter搜索...")
            all_docs = vectorstore.similarity_search(
                query=f"paper_id:{paper_id}", 
                k=5
            )
            print(f"不使用filter找到 {len(all_docs)} 个文档")
            if all_docs:
                print("所有文档的metadata:")
                for i, doc in enumerate(all_docs):
                    print(f"  文档 {i+1}: {doc.metadata}")
    else:
        print("没有获取到任何文档")


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--papers_mineru_dir', type=str, default='./export/papers_mineru', help='The path to the papers mineru directory')
    args.add_argument('--chroma_dir', type=str, default='./export/chroma', help='The path to the chroma directory')
    args.add_argument('--db_path', type=str, default='./export/db/academy.db', help='The path to the database file')
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