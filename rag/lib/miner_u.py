from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

import argparse
import os
import subprocess
import time

def run_mineru(file_path, output_dir):
    """
    使用subprocess确保命令执行完成后再返回。
    """
    try:
        cmd = f"mineru -m txt -b pipeline -p \"{file_path}\" -o \"{output_dir}\" -l en"
        print(cmd)
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print(f"mineru 执行失败，返回码: {result.returncode}")
    except Exception as e:
        print(f"mineru 执行失败，错误: {e}")


def main(root_dir, input_dir, output_dir):
    # 获取所有 PDF 文件
    input_path = os.path.join(root_dir, input_dir)
    pdf_files = [file for file in os.listdir(input_path) if file.endswith('.pdf')]
    pdf_files = [file for file in pdf_files if file.replace('.pdf', '') not in os.listdir(os.path.join(root_dir, output_dir))]
    
    print(f"找到 {len(pdf_files)} 个 PDF 文件需要处理")
    
    if not pdf_files:
        print("没有需要处理的文件")
        return
    
    # 串行处理每个文件
    try:
        for idx, file in enumerate(pdf_files, 1):
            file_path = os.path.join(root_dir, input_dir, file)
            output_path = os.path.join(root_dir, output_dir)
            
            print(f"\n正在处理第 {idx}/{len(pdf_files)} 个文件: {file}")
            
            try:
                run_mineru(file_path, output_path)
                print(f"  ✓ 完成: {file}")
            except Exception as e:
                print(f"  ✗ 失败: {file}, 错误: {e}")
            
            # 在处理文件之间添加小延迟，让系统休息一下
            if idx < len(pdf_files):
                print("等待 1 秒后继续下一个文件...")
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("\n收到中断信号，正在停止...")
        return
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
    
    print("所有任务已完成")


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--root_dir', type=str, default='/Users/kexu/Library/CloudStorage/OneDrive-Personal/Academy', help='The root directory of the input')
    args.add_argument('--input_dir', type=str, default='papers', help='The directory of the input')
    args.add_argument('--output_dir', type=str, default='papers_mineru', help='The directory of the output')
    args = args.parse_args()
    
    root_dir = args.root_dir
    input_dir = args.input_dir
    output_dir = args.output_dir

    main(root_dir, input_dir, output_dir)