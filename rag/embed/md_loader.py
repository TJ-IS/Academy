from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

import argparse
import os
import json
from langchain_text_splitters import MarkdownHeaderTextSplitter


def main(args):
    headers_to_split_on = [
        ('#', 'Section'),
        ('##', 'Subsection'),
        ('###', 'Subsubsection'),
    ]

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    for paper_dir in os.listdir(args.papers_mineru_dir):
        for file in os.listdir(os.path.join(args.papers_mineru_dir, paper_dir, 'txt')):
            if file.endswith('.md'):
                file_path = os.path.join(args.papers_mineru_dir, paper_dir, 'txt', file)
                with open(file_path, 'r') as f:
                    md_content = f.read()
        break

    md_header_splits = splitter.split_text(md_content)

    with open('./test.txt', 'w') as f:
        for doc in md_header_splits:
            print(doc.metadata)
            f.write(doc.page_content)
            f.write('\n')


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--papers_mineru_dir', type=str, default='./export/papers_mineru', help='The path to the md directory')
    args = args.parse_args()

    if not os.path.exists(args.papers_mineru_dir):
        print(f'[ERROR] Papers mineru directory {args.papers_mineru_dir} does not exist')
        exit(1)

    main(args)
    
    