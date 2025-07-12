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


def get_title_by_paper_id(paper_id, args) -> str:
    try:
        conn = sqlite3.connect(args.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT title FROM paper WHERE id = ?", (paper_id,))
        title = cursor.fetchone()[0]
        conn.close()
        return title
    except Exception as e:
        print(f'[ERROR] Failed to get title by paper id {paper_id}: {e}')
        return ''


def transform_md_to_docs(md_header_splits, paper_dir, args) -> List[Document]:
    docs = []
    paper_id = paper_dir 
    paper_title = get_title_by_paper_id(paper_dir, args)
    
    for doc in md_header_splits:
        docs.append(Document(
            id = str(uuid.uuid4()),
            page_content = doc.page_content,
            metadata = {
                'paper_id': int(paper_id),
                'paper_title': paper_title,
                'section': doc.metadata['section'],
            }
        ))

    return docs


def main(args):
    headers_to_split_on = [
        ('#', 'section'),
    ]

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    for paper_dir in os.listdir(args.papers_mineru_dir):
        for file in os.listdir(os.path.join(args.papers_mineru_dir, paper_dir, 'txt')):
            if file.endswith('.md'):
                file_path = os.path.join(args.papers_mineru_dir, paper_dir, 'txt', file)
                with open(file_path, 'r') as f:
                    md_content = f.read()
                    md_header_splits = splitter.split_text(md_content)
                    docs = transform_md_to_docs(md_header_splits, paper_dir, args)
                    with open('./test.txt', 'w') as f:
                        for doc in docs:
                            f.write(json.dumps(doc.model_dump(), indent=4))
                            f.write('\n')

        break



if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--papers_mineru_dir', type=str, default='./export/papers_mineru', help='The path to the md directory')
    args.add_argument('--db_path', type=str, default='./export/db/academy.db', help='The path to the database file')
    args = args.parse_args()

    if not os.path.exists(args.papers_mineru_dir):
        print(f'[ERROR] Papers mineru directory {args.papers_mineru_dir} does not exist')
        exit(1)

    main(args)
    
    