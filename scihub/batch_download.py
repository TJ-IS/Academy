import os
from scidownl import scihub_download
from tqdm import tqdm
import pandas as pd
import argparse


def main(scopus_file_path, export_dir):
    df = pd.read_csv(scopus_file_path)

    for i, row in tqdm(df.iterrows(), total=len(df)):
        doi = row['DOI']
        year = row['年份']
        title = row['文献标题']
        
        if not isinstance(doi, str) or not isinstance(title, str):
            continue

        file_name = f'{title.replace("/", "_")}.pdf'
        
        # 如果文件已下载，则跳过
        if os.path.exists(os.path.join(export_dir, file_name)):
            continue
        
        # 如果文件不存在，则下载
        try: 
            scihub_download(doi, out=os.path.join(export_dir, file_name), paper_type='doi')
        except Exception as e:
            print(e)
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scopus_file_path', type=str, default='/Volumes/YMTC-512G/Projects/Academy/data/scopus/ais_2010_2015.csv', help='The path to the scopus csv file')
    parser.add_argument('--export_dir', type=str, default='/Volumes/YMTC-512G/Projects/Academy/export/ais_2010_2015/', help='The path to the export directory')
    args = parser.parse_args()

    if not os.path.exists(args.scopus_file_path):
        print(f'File {args.scopus_file_path} does not exist')
        exit(1)

    os.makedirs(args.export_dir, exist_ok=True)

    main(args.scopus_file_path, args.export_dir)