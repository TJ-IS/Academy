import os
import sqlite3
import shutil
import argparse

def main(args):
    conn = sqlite3.connect(args.db_path)
    cursor = conn.cursor()

    for pdf_dir in args.pdf_dirs:
        if not os.path.exists(pdf_dir):
            print(f'[WARNING] Pdf directory {pdf_dir} does not exist')
            continue
        else:
            for file in os.listdir(pdf_dir):
                if file.endswith('.pdf'):
                    file_path = os.path.join(pdf_dir, file)
                else: continue
                
                # Get paper info
                cursor.execute('''
                    SELECT * FROM paper WHERE title = ?
                ''', (file.replace('.pdf', ''),))
                paper_info = cursor.fetchone()
                
                # Copy file
                if paper_info is not None:
                    paper_id = paper_info[0]
                    export_file_path = os.path.join(args.export_dir, f'{paper_id}.pdf')
                    shutil.copy(file_path, export_file_path)
                    cursor.execute('''
                        UPDATE paper SET file_exists = 1 WHERE id = ?
                    ''', (paper_id,))
                    conn.commit()
                else: continue


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--db_path', type=str, default='./export/db/academy.db', help='The path to the database file')
    args.add_argument('--pdf_dirs', type=list, default=['./export/ais_2015_2020', './export/ais_2010_2015', './export/ais_2005_2010'], help='The paths to the pdf directories')
    args.add_argument('--export_dir', type=str, default='papers', help='The path to the export directory')
    args = args.parse_args()

    os.makedirs(args.export_dir, exist_ok=True)

    main(args)