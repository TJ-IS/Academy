import os
import re
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
import numpy as np

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from rag.embed.md_loader import get_paper_md


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


# Define the output structure of the Pydantic model
class PaperInfo(BaseModel):
    keywords: List[str] = Field(description="The list of keywords in the paper")
    summary: str = Field(description="The summary of the paper, including the main contributions, methodology, and conclusions")
    content: List[HypothesisOrResearchQuestion] = Field(description="The content of the paper, which can be hypotheses or research questions.")


async def chat(paper_id, text, args):
    llm = ChatOpenAI(model=args.model, temperature=0)
    
    # Create output parser
    parser = PydanticOutputParser(pydantic_object=PaperInfo)
    format_instructions = parser.get_format_instructions()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You're a PHD student in Information Systems.

# Goal
Your task is to extract paper information from the transcript of the academic paper in the field of Information Systems. You will be given a transcript of the paper. You need to follow the guidelines and constraints to extract the information.

# Guidelines
1. First, you need to focus on paper's ABSTRACT section to understand the main content of the paper.
2. Second, you need to distinguish what is the main research question or problem of the paper.
3. Third, if the paper contains hypotheses or research questions, you need to understand them, and find out the independent variable and dependent variable in the hypotheses or research questions.
4. Fourth, you need to extract the keywords and the summary of the paper.

- Some papers may contain hypotheses or research questions, you need to extract them. 
- While you are extracting the hypotheses or research questions, you need to extract the independent variable and dependent variable that are used in the hypotheses or research questions. 
- If the iv or dv is multiple, you need to extract them all, and separate them with commas.
- The iv_description and dv_description should be the description of the independent variable and dependent variable.


# Constraints
1. Do not generate any other text or comments that are not included in the transcript.
2. You need to extract the following information from the transcript: 
- Keywords
- Summary
- Content

# Output Format
You will need to return the information in the following JSON format:
{format_instructions}
         """),
        ("human", "Below is the transcript of the paper:\n\n{text}"),
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    try:
        result = await chain.ainvoke({
            "text": text,
            "format_instructions": format_instructions
        })
    except Exception as e:
        print(f'[ERROR] Failed to extract the paper {paper_id}: {e}')
        return None

    return result


async def get_the_main_content(paper_md):
    """
    Get the main content of the paper, which is the content before the References section.
    """
    pattern = r'^\s*#+?\s*(?:R\s*E\s*F\s*E\s*R\s*E\s*N\s*C\s*E\s*S?|R\s*E\s*F|REFERENCES?|References?|Ref)\s*.*$'
    
    match = re.search(pattern, paper_md, re.MULTILINE | re.IGNORECASE)
    
    if match:
        paper_md = paper_md[:match.start()].rstrip()
    
    return paper_md


async def process_paper_by_id(paper_id, args, semaphore, delay_seconds):
    if os.path.exists(os.path.join(args.output_dir, f'{paper_id}.json')):
        return None

    await asyncio.sleep(delay_seconds)
    async with semaphore:
        paper_md = get_paper_md(paper_id, args)
        paper_md = await get_the_main_content(paper_md)
        paper_info = await chat(paper_id, paper_md, args)

        try:
            if paper_info:
                paper_info = repair_json(paper_info)
                paper_info = json.loads(paper_info)
        except Exception as e:
            print(f"[ERROR] Failed to parse the paper info for paper {paper_id}: {e}")
            await asyncio.sleep(3)
            return None

        if paper_info:
            with open(os.path.join(args.output_dir, f'{paper_id}.json'), 'w', encoding='utf-8') as f:
                json.dump(paper_info, f, indent=2)

    return paper_info


async def main():
    semaphore = asyncio.Semaphore(10)

    paper_ids = [i for i in os.listdir(args.papers_mineru_dir) if os.path.isdir(os.path.join(args.papers_mineru_dir, i))]
    paper_ids = sorted(paper_ids)

    batch_size = 200
    for batch_idx in range(0, len(paper_ids), batch_size):
        print(f"Processing batch {batch_idx} of {len(paper_ids)}")
        batch_paper_ids = paper_ids[batch_idx:min(batch_idx+batch_size, len(paper_ids))]
        delay_seconds = np.random.exponential(scale=10, size=len(batch_paper_ids))
        tasks = [process_paper_by_id(paper_id, args, semaphore, delay_seconds[i]) for i, paper_id in enumerate(batch_paper_ids)]
        await tqdm_asyncio.gather(*tasks, desc="Processing papers", unit="paper")

    
async def dev():
    semaphore = asyncio.Semaphore(1)
    paper_info = await process_paper_by_id(10, args, semaphore, 0)
    if paper_info:
        print(json.dumps(paper_info, indent=2))
        with open(os.path.join(args.output_dir, f'{10}.json'), 'w', encoding='utf-8') as f:
            json.dump(paper_info, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--papers_mineru_dir', type=str, default='./export/papers_mineru', help='The path to the md directory')
    parser.add_argument('--output_dir', type=str, default='./export/papers_info', help='The path to the output directory')
    parser.add_argument('--db_path', type=str, default='./export/db/academy.db', help='The path to the database file')
    parser.add_argument('--model', type=str, default='gemini-2.5-flash-all', help='The model to use')
    parser.add_argument('--dev', action='store_true', help='Run in development mode')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.dev:
        asyncio.run(dev())
    else:
        asyncio.run(main())