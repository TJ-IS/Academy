from dotenv import load_dotenv, find_dotenv
from pyarrow.compute import docscrape
_ = load_dotenv(find_dotenv())

import argparse
import os
import json
import sqlite3
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
import uuid
import pickle
from tqdm import tqdm


def get_paper_title_journal_year(paper_id, args) -> tuple[str, str, int]:
    """Get the title, journal, and year of a paper"""
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


def get_paper_md(paper_id, args):
    """Get the markdown content of a paper"""
    paper_md_path = os.path.join(args.papers_mineru_dir, str(paper_id), 'txt', f'{paper_id}.md')
    if not os.path.exists(paper_md_path):
        return ''
    with open(paper_md_path, 'r') as f:
        return f.read()


def get_paper_docs(paper_id, args):
    """Get Documents from a paper, only split by section"""
    paper_md = get_paper_md(paper_id, args)
    if not paper_md:
        return []
    else:
        paper_title, paper_journal, paper_year = get_paper_title_journal_year(paper_id, args)

    headers_to_split_on = [
        ('#', 'section'),
    ]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = splitter.split_text(paper_md)

    docs = []
    for md_header_split in md_header_splits:
        docs.append(Document(
            id = str(uuid.uuid4()),
            page_content = md_header_split.page_content,
            metadata = {
                'paper_id': str(paper_id),
                'paper_title': paper_title.title(),
                'paper_journal': paper_journal.title(),
                'paper_year': str(paper_year),
                'section': prettier_section(md_header_split.metadata['section']) if 'section' in md_header_split.metadata else '',
            },
        ))

    return docs


def get_paper_docs_recursive(paper_id, args):
    """Get Documents from a paper recursively"""
    paper_title, paper_journal, paper_year = '', '', ''
    paper_md = get_paper_md(paper_id, args)
    if not paper_md:
        return []
    else:
        paper_title, paper_journal, paper_year = get_paper_title_journal_year(paper_id, args)
    
    headers_to_split_on = [
        ('#', 'section'),
    ]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = splitter.split_text(paper_md)

    docs = []

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    for md_header_split in md_header_splits:
        sub_docs = recursive_splitter.split_text(md_header_split.page_content)
        for sub_doc in sub_docs:
            docs.append(Document(
                id = str(uuid.uuid4()),
                page_content = sub_doc,
                metadata = {
                    'paper_id': str(paper_id),
                    'paper_title': paper_title.title(),
                    'paper_journal': paper_journal.title(),
                    'paper_year': str(paper_year),
                    'section': prettier_section(md_header_split.metadata['section']) if 'section' in md_header_split.metadata else '',
                },
            ))
    
    return docs


def prettier_section(section):
    new_section = section

    # General cases
    if 'introduction' in section.lower() or 'introduction' in section.lower().strip(' '):
        new_section = 'Introduction'
    elif 'conclusion' in section.lower() or 'conclusion' in section.lower().strip(' '):
        new_section = 'Conclusion'
    elif 'discussion' in section.lower() or 'discussion' in section.lower().strip(' '):
        new_section = 'Discussion'
    elif 'results' in section.lower() or 'results' in section.lower().strip(' '):
        new_section = 'Results'
    elif 'methods' in section.lower() or 'methods' in section.lower().strip(' '):
        new_section = 'Methods'
    elif 'reference' in section.lower() or 'reference' in section.lower().strip(' '):
        new_section = 'Reference'

    return new_section.title()


def get_all_sections(args):
    """Get all sections from all papers in mineru directory"""
    paper_ids = [i for i in os.listdir(args.papers_mineru_dir) if os.path.isdir(os.path.join(args.papers_mineru_dir, i))]
    sections = []
    for paper_id in tqdm(paper_ids):
        docs = get_paper_docs_recursive(paper_id, args)
        for doc in docs:
            sections.append(prettier_section(doc.metadata['section']))

    return sections


def main(args):
    pass


def dev(args):
    paper_id = '1'
    docs = get_paper_docs(paper_id, args)
    print(f'get_paper_docs: {len(docs)}')
    docs = get_paper_docs_recursive(paper_id, args)
    print(f'get_paper_docs_recursive: {len(docs)}')

    # sections = get_all_sections(args)
    # with open(os.path.join('temp', 'sections.txt'), 'w') as f:
    #     for section in list(set(sections)):
    #         f.write(f'{section}\n')


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