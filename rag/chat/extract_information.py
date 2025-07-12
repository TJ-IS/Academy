import os
import asyncio
import json
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional, Union
from json_repair import repair_json
from tqdm.asyncio import tqdm_asyncio
import argparse


class HypothesisOrResearchQuestion(BaseModel):
    type: str = Field(description="The type of the hypothesis or research question, which can be 'hypothesis' or 'research question'.")
    description: str = Field(description="The description of the hypothesis or research question.")
    iv: str = Field(description="The independent variable of the hypothesis.")
    iv_description: str = Field(description="The description of the independent variable.")
    dv: str = Field(description="The dependent variable of the hypothesis or research question.")
    dv_description: str = Field(description="The description of the dependent variable.")
    method: str = Field(description="The method of the hypothesis or research question.")
    result: str = Field(description="The result of the hypothesis or research question.")
    conclusion: str = Field(description="The conclusion of the hypothesis or research question.")


# 定义输出结构的 Pydantic 模型
class PaperInfo(BaseModel):
    keywords: List[str] = Field(description="The list of keywords in the paper")
    summary: str = Field(description="The summary of the paper, including the main contributions, methodology, and conclusions")
    content: List[HypothesisOrResearchQuestion] = Field(description="The content of the paper, which can be hypotheses or research questions.")


data_dir = r'/Users/kexu/Library/CloudStorage/OneDrive-Personal/Academy'
input_dir = 'papers'
output_dir = 'papers_info'
if not os.path.exists(os.path.join(data_dir, output_dir)):
    os.makedirs(os.path.join(data_dir, output_dir))

def get_all_txt_files(data_dir, sub_dir, skipping_existing=False):
    txt_files = [f for f in os.listdir(os.path.join(data_dir, sub_dir)) if f.endswith('.txt')]
    if skipping_existing:
        txt_files = [f for f in txt_files if f.replace('.txt', '.json') not in os.listdir(os.path.join(data_dir, output_dir))]
    return txt_files


async def chat(text):
    llm = ChatOpenAI(model="gemini-2.5-flash-all", temperature=0)
    
    # 创建输出解析器
    parser = PydanticOutputParser(pydantic_object=PaperInfo)
    format_instructions = parser.get_format_instructions()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
         # Task
         You are a helpful assistant that extracts information from a academic paper.
         You will be given a transcript of the paper.
         
         # Requirements
         You will need to extract the following information from the transcript:
         - Keywords
         - Summary
         - Content
         
         # Guidelines
         - The keywords should be the most important words in the paper.
         - The summary should be a concise summary of the paper.
         - The content should be the content of the paper. 
         - Some papers may contain hypotheses or research questions, you need to extract them. 
         - If the paper contains both hypotheses and research questions, you need to extract them both. 
         - If the paper contains neither hypotheses nor research questions, you need to return an empty string.
         - If the paper contains only hypothesis or research question, you need to return the hypothesis or research question, and not to return the one that is not in the paper.
         - While you are extracting the hypotheses or research questions, you need to extract the independent variable and dependent variable that are used in the hypotheses or research questions. 
         - If the iv or dv is multiple, you need to extract them all, and separate them with commas.
         - The iv_description and dv_description should be the description of the independent variable and dependent variable.
         - Do not generate any other text or comments that are not included in the transcript.
         
         # Output Format
         You will need to return the information in the following JSON format:
         {format_instructions}
         """),
        ("human", "Here is the transcript of the paper:\n\n{text}"),
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    return await chain.ainvoke({
        "text": text,
        "format_instructions": format_instructions
    })


async def process_single_file(txt_file, semaphore):
    """处理单个文件的异步函数"""
    async with semaphore:  # 控制并发数量
        if os.path.exists(os.path.join(data_dir, output_dir, txt_file.replace('.txt', '.json'))):
            # print(f'{txt_file} already exists')  # 减少输出干扰进度条
            return f'{txt_file} (已存在)'
        
        try:
            with open(os.path.join(data_dir, input_dir, txt_file), 'r', encoding='utf-8') as f:
                text = f.read()
                
                # 调用聊天函数获取结构化输出
                result = await chat(text)
                
                # 处理结果
                result = repair_json(result)
                result = json.loads(result)
                
                # 保存结果
                with open(os.path.join(data_dir, output_dir, txt_file.replace('.txt', '.json')), 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                
                return f'{txt_file} (已完成)'
                
        except Exception as e:
            return f'{txt_file} (错误: {e})'


async def main():
    # 创建信号量，限制最多同时处理3个文件
    semaphore = asyncio.Semaphore(args.num_workers)
    
    print(f"开始处理 {len(txt_files)} 个文件...")
    print("-" * 50)
    
    # 创建所有文件处理任务
    tasks = [process_single_file(txt_file, semaphore) for txt_file in txt_files]
    
    # 使用 tqdm 显示进度并发执行所有任务
    results = await tqdm_asyncio.gather(
        *tasks, 
        desc="处理文件进度",
        unit="file"
    )
    
    print(f"\n所有 {len(txt_files)} 个文件处理完成！")


async def dev():
    txt_file = txt_files[0]
    with open(os.path.join(data_dir, input_dir, txt_file), 'r', encoding='utf-8') as f:
        text = f.read()
        
        result = await chat(text)
        print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=10, help='Number of workers to process files')
    parser.add_argument('--skipping_existing', action='store_true', help='Skip existing files')
    parser.add_argument('--dev', action='store_true', help='Run in development mode')
    args = parser.parse_args()
    
    txt_files = get_all_txt_files(data_dir, input_dir, args.skipping_existing)
    
    if args.dev:
        asyncio.run(dev())
    else:
        asyncio.run(main())