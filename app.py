from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import openai
from fastapi.middleware.cors import CORSMiddleware


load_dotenv()

embedding_function = OpenAIEmbeddings()


# docs = [
#     Document(
#         page_content="the dog loves to eat pizza", metadata={"source": "animal.txt"}
#     ),
#     Document(
#         page_content="the cat loves to eat lasagna", metadata={"source": "animal.txt"}
#     ),
# ]

# db = Chroma.from_documents(docs, embedding_function)
# retriever = db.as_retriever()


template = """Answer the question:

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI(model="gpt-4o-mini",temperature=0, streaming=True)

retrieval_chain = (
    {"question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace '*' with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def generate_chat_responses(message):
    async for chunk in retrieval_chain.astream(message):
        content = chunk.replace("\n", "<br>")
        yield f"data: {content}\n\n"


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/chat_stream/{message}")
async def chat_stream(message: str):
    return StreamingResponse(generate_chat_responses(message=message), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
