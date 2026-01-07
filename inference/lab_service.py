# inference/lab_service.py
from fastapi import FastAPI, File, UploadFile
import torch
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI()

# Connect to your local vLLM serving Qwen
llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token",
    model="Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4"
)

@app.post("/ocr")
async def handle_ocr(file: UploadFile = File(...)):
    # 1. Save temp PDF
    temp_pdf = Path(f"temp_{file.filename}")
    with open(temp_pdf, "wb") as buffer:
        buffer.write(await file.read())

    # 2. Run your existing OCR Logic (TrOCR + CRAFT)
    # raw_text = run_your_ocr_logic(temp_pdf)
    
    # 3. Agentic Correction via LangChain
    prompt = ChatPromptTemplate.from_template(
        "Clean this noisy OCR and return ONLY a JSON object with "
 ## inference/lab_service.py
from fastapi import FastAPI, File, UploadFile
import torch
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI()

# Connect to your local vLLM serving Qwen
llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token",
    model="Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4"
)

@app.post("/ocr")
async def handle_ocr(file: UploadFile = File(...)):
    # 1. Save temp PDF
    temp_pdf = Path(f"temp_{file.filename}")
    with open(temp_pdf, "wb") as buffer:
        buffer.write(await file.read())

    # 2. Run your existing OCR Logic (TrOCR + CRAFT)
    # raw_text = run_your_ocr_logic(temp_pdf)
    
    # 3. Agentic Correction via LangChain
    prompt = ChatPromptTemplate.from_template(
        "Clean this noisy OCR and return ONLY a JSON object with "
        "keys 'main_topic' and 'keywords':\n{text}"
    )
    chain = prompt | llm
    ai_msg = chain.invoke({"text": "example noisy text"})
    
    return {
        "filename": file.filename,
        "structured_data": ai_msg.content
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) inference/lab_service.py
from fastapi import FastAPI, File, UploadFile
import torch
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI()

# Connect to your local vLLM serving Qwen
llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token",
    model="Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4"
)

@app.post("/ocr")
async def handle_ocr(file: UploadFile = File(...)):
    # 1. Save temp PDF
    temp_pdf = Path(f"temp_{file.filename}")
    with open(temp_pdf, "wb") as buffer:
        buffer.write(await file.read())

    # 2. Run your existing OCR Logic (TrOCR + CRAFT)
    # raw_text = run_your_ocr_logic(temp_pdf)
    
    # 3. Agentic Correction via LangChain
    prompt = ChatPromptTemplate.from_template(
        "Clean this noisy OCR and return ONLY a JSON object with "
        "keys 'main_topic' and 'keywords':\n{text}"
    )
    chain = prompt | llm
    ai_msg = chain.invoke({"text": "example noisy text"})
    
    return {
        "filename": file.filename,
        "structured_data": ai_msg.content
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)       "keys 'main_topic' and 'keywords':\n{text}"
    )
    chain = prompt | llm
    ai_msg = chain.invoke({"text": "example noisy text"})
    
    return {
        "filename": file.filename,
        "structured_data": ai_msg.content
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
