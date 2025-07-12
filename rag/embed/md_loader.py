from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

import argparse
import os
import json
import sqlite3
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from typing import List
import uuid
import pickle
from tqdm import tqdm


def get_title_by_paper_id(paper_id, args) -> tuple[str, str, int]:
    title, journal, year = '', '', -1
    try:
        conn = sqlite3.connect(args.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT title, journal, year FROM paper WHERE id = ?", (paper_id,))
        result = cursor.fetchone()
        if result:
            title, journal, year = result
        conn.close()
        return title, journal, year
    except Exception as e:
        print(f'[ERROR] Failed to get title by paper id {paper_id}: {e}')
    return title, journal, year


def transform_md_to_docs(md_header_splits, paper_dir, args) -> List[Document]:
    docs = []
    paper_id = paper_dir 
    paper_title, paper_journal, paper_year = get_title_by_paper_id(paper_dir, args)
    
    for doc in md_header_splits:
        docs.append(Document(
            id = str(uuid.uuid4()),
            page_content = doc.page_content,
            metadata = {
                'paper_id': str(paper_id),  # 改为字符串类型以兼容Chroma filter
                'paper_title': paper_title,
                'paper_journal': paper_journal,
                'paper_year': str(paper_year),  # 改为字符串类型
                'section': doc.metadata['section'] if 'section' in doc.metadata else '',
            }
        ))

    return docs


def get_paper_md(paper_id, args):
    paper_md_path = os.path.join(args.papers_mineru_dir, str(paper_id), 'txt', f'{paper_id}.md')
    if not os.path.exists(paper_md_path):
        return ''
    with open(paper_md_path, 'r') as f:
        return f.read()


def get_paper_docs(paper_id, args):
    paper_md = get_paper_md(paper_id, args)
    if not paper_md:
        return []

    headers_to_split_on = [
        ('#', 'section'),
    ]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = splitter.split_text(paper_md)
    return transform_md_to_docs(md_header_splits, paper_id, args)


def main(args):
    headers_to_split_on = [
        ('#', 'section'),
    ]

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    for paper_dir in tqdm(os.listdir(args.papers_mineru_dir)):
        for file in os.listdir(os.path.join(args.papers_mineru_dir, paper_dir, 'txt')):
            if file.endswith('.md'):
                file_path = os.path.join(args.papers_mineru_dir, paper_dir, 'txt', file)
                with open(file_path, 'r') as f:
                    md_content = f.read()
                    md_header_splits = splitter.split_text(md_content)
                    docs = transform_md_to_docs(md_header_splits, paper_dir, args)
                    
                    pickle.dump(docs, open(os.path.join(args.output_dir, f'{paper_dir}.pkl'), 'wb'))
                    # with open(os.path.join(args.output_dir, f'{paper_dir}.txt'), 'w') as f:
                    #     for doc in docs:
                    #         f.write(json.dumps(doc.model_dump(), indent=4))
                    #         f.write('\n')

        # break

def dev(args):
    paper_docs = get_paper_docs(1, args)
    print(paper_docs)



if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--papers_mineru_dir', type=str, default='./export/papers_mineru', help='The path to the md directory')
    args.add_argument('--db_path', type=str, default='./export/db/academy.db', help='The path to the database file')
    args.add_argument('--output_dir', type=str, default='./export/papers_docs', help='The path to the output directory')
    args.add_argument('--dev', action='store_true', help='Use dev mode')
    args = args.parse_args()

    if not os.path.exists(args.papers_mineru_dir):
        print(f'[ERROR] Papers mineru directory {args.papers_mineru_dir} does not exist')
        exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.dev:
        dev(args)
    else:
        main(args)
    
    