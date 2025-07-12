import sqlite3
import os
import argparse


def main(args):
    conn = sqlite3.connect(args.db_path)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM paper
    ''')
    papers = cursor.fetchall()
    print(f'Total papers: {len(papers)}')
    print(f'Total papers with file: {len([paper for paper in papers if paper[6] == 1])}')
    print(f'Total papers without file: {len([paper for paper in papers if paper[6] == 0])}')


    min_year = min([paper[3] for paper in papers])
    max_year = max([paper[3] for paper in papers])
    print(f'Min year: {min_year}')
    print(f'Max year: {max_year}')


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--db_path', type=str, default='./export/db/academy.db', help='The path to the database file')
    args = args.parse_args()

    if not os.path.exists(args.db_path):
        print(f'[ERROR] Database file {args.db_path} does not exist')
        exit(1)
    else:
        main(args)