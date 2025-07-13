from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

import json_repair
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional, Union
from tqdm.asyncio import tqdm_asyncio
import argparse
import asyncio
import json

def get_paper_header_structure(paper_docs):
    header_structure = []
    for doc in paper_docs:
        header_structure.append(doc.metadata['section'])
    return header_structure


async def prettier_header_structure(header_structure):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""
# Goal
Your task is to add a markdown header prefix to the headers in the list.

You're providing with a list of headers extracted from a pdf loader, which might be in an irregular format. More specifically, the level of the header might be inconsistent.
Your task is to add a markdown header prefix to the headers in the list, and decide the level of the header. The maximum level of the header is 3. (e.g. "" "#" "##" or "###").

# Constraints
- First, you should decide whether the header is a header or not. If it is not a header, then just return the original string and do not add any prefix.
- Second, you should decide the level of the header. The maximum level of the header is 3. (e.g. "#" "##" or "###").
- Third, you need to convert the header string in a unified format. In this step, you should follow the following rules:
    1. Capital Style: The capitalization of the header should be consistent with the SENTENCE CASE.
    2. Remove the number prefix: If there is no such label as "1.1" in the header, then judge independently according to the context what level it should belong to.
    3. Remove the number prefix: If there is such label as "1." or "1.1" in the header, then delete these labels.
    4. If the header is an empty string, then just return the empty string. Do not ignore it.
    5. If there are multiple headers in the same item, then you should convert them to the same item but with different levels. For example, "3. Empirical Analysis 3.1. Data" -> "# Empirical Analysis\n\n## Data" (Do not separate them into multiple items in the response format)

# Few shot examples
"A B S T R A C T" -> "# Abstract"
"1. Introduction" -> "# Introduction"
"1.1. Background" -> "## Background"
"1.1.1. Research Question" -> "### Research question"
"1.1.2. Research Objectives" -> "### Research objectives"
"1.1.3. Research Methodology" -> "### Research methodology"
"1.1.4. Research Findings" -> "### Research findings"
"3. Empirical Analysis 3.1. Data" -> "# Empirical Analysis ## Data"

# Response format
- The response should be a list of headers in markdown format.
- The number of headers should be consistent with the original.

[EXAMPLE_INPUT]
```json
['', 'Information Systems Research', 'How Does the Mobile Channel Reshape the Sales Distribution in E-Commerce?', '1. Introduction', '2. Search Affordances and Constraints of the Mobile Channel', '3. Empirical Analysis 3.1. Data', '3.2. Comparison of the Sales Distribution Between the PC and Mobile Channels', '3.3. Impacts of Mobile Channel Adoption on the Search Intensity and Sales Distribution', '4. Discussion and Conclusions', 'Acknowledgments', 'Appendix A. Plots of the Dependent Variables of PC-Mobile and PC-Only Buyers', 'Appendix C. Analysis Results with the First Mobile Purchase as Treatment', 'Endnotes']
```
[EXAMPLE_OUTPUT]
```json
["# Information Systems Research", "# How Does the Mobile Channel Reshape the Sales Distribution in E-Commerce?", "# Introduction", "# Search Affordances and Constraints of the Mobile Channel", "# Empirical Analysis ## Data", "# Comparison of the Sales Distribution Between the PC and Mobile Channels", "# Impacts of Mobile Channel Adoption on the Search Intensity and Sales Distribution", "# Discussion and Conclusions", "# Acknowledgments", "# Appendix A. Plots of the Dependent Variables of PC-Mobile and PC-Only Buyers", "# Appendix C. Analysis Results with the First Mobile Purchase as Treatment", "# Endnotes"]
```
        """),
        ("user", "{header_structure}"),
    ])
    chain = prompt | llm | StrOutputParser()
    result = await chain.ainvoke({
        "header_structure": json.dumps(header_structure, ensure_ascii=False)
    })

    result = json_repair.repair_json(result)

    return json.loads(result)
    

async def main(args):
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    from rag.embed.md_loader import get_paper_docs
    paper_docs = get_paper_docs(10, args)
    header_structure = get_paper_header_structure(paper_docs)

    print(header_structure)
    print('-' * 100)

    header_structure = await prettier_header_structure(header_structure)

    print(header_structure)



if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--papers_mineru_dir', type=str, default='./export/papers_mineru', help='The path to the md directory')
    args.add_argument('--db_path', type=str, default='./export/db/academy.db', help='The path to the database file')
    args = args.parse_args()


    asyncio.run(main(args))