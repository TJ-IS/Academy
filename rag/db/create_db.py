import os
import argparse
import sqlite3
import pandas as pd
from tqdm import tqdm


class PaperInfo:
    title: str
    doi: str
    year: int
    authors: str
    journal: str
    file_exists: bool


def create_db(args):
    if not os.path.exists(args.db_path):
        open(args.db_path, 'a').close()
    
    conn = sqlite3.connect(args.db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS paper (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            doi TEXT,
            year INTEGER,
            authors TEXT,
            journal TEXT,
            file_exists BOOLEAN
        )
    ''')
    conn.commit()
    conn.close()


def insert_paper(args, paper_info: PaperInfo):
    try:
        conn = sqlite3.connect(args.db_path)
        # 通过文献标题查询文献是否存在
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM paper WHERE title = ?
        ''', (paper_info.title,))
        if cursor.fetchone() is not None:
            print(f'Title {paper_info.title} already exists')
            return
        
        # 通过文献 DOI 查询文献是否存在
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM paper WHERE doi = ?
        ''', (paper_info.doi,))
        if cursor.fetchone() is not None:
            print(f'DOI {paper_info.doi} already exists')
            return
        
        # 插入文献信息
        cursor.execute('''
            INSERT INTO paper (title, doi, year, authors, journal, file_exists) VALUES (?, ?, ?, ?, ?, ?)
        ''', (paper_info.title, paper_info.doi, paper_info.year, paper_info.authors, paper_info.journal, paper_info.file_exists))
        conn.commit()
    except Exception as e:
        print(f'[ERROR] {e}')
    finally:
        conn.close()


def main(args):
    print('[INFO] Creating database...')
    if args.init:
        create_db(args)
    else:
        print('[INFO] Inserting papers...')
        for scopus_csv in args.scopus_csvs:
            print(f'[INFO] Processing {scopus_csv}...')
            if not os.path.exists(scopus_csv):
                print(f'[WARNING] Scopus csv {scopus_csv} does not exist')
                continue
            else:
                df = pd.read_csv(scopus_csv)
            for index, row in tqdm(df.iterrows(), total=len(df)):
                paper_info = PaperInfo()
                paper_info.title = str(row['文献标题'])
                paper_info.doi = str(row['DOI'])
                paper_info.year = int(row['年份'])
                paper_info.authors = str(row['作者'])
                paper_info.journal = str(row['来源出版物名称'])
                paper_info.file_exists = False
                insert_paper(args, paper_info)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--scopus_csvs', type=list, default=['./data/scopus/ais_2005_2010.csv', './data/scopus/ais_2010_2015.csv', './data/scopus/ais_2015_2020.csv'], help='The paths to the scopus csv files')
    args.add_argument('--db_dir', type=str, default='./export/db', help='The path to the database directory')
    args.add_argument('--db_file', type=str, default='academy.db', help='The name of the database file')
    args.add_argument('--init', action='store_true', help='Initialize the database')
    args = args.parse_args()

    os.makedirs(args.db_dir, exist_ok=True)
    args.db_path = os.path.join(args.db_dir, args.db_file)

    for scopus_csv in args.scopus_csvs:
        if not os.path.exists(scopus_csv):
            print(f'[WARNING] Scopus csv {scopus_csv} does not exist')
    
    main(args)