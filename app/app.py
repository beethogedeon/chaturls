from fastapi import FastAPI
from helper import Model
from typing import Optional, List
from pydantic import BaseModel
import torch
import os

app = FastAPI(
    title="Q/A Chatbot",
    description="A Q/A chatbot trained with data from provided websites.",
    version="0.1.0",
    docs_url="/",
    redoc_url=None
)

if os.path.exists("../models/latest.pt"):
    model = torch.load("../models/latest.pt")


class Api_key(BaseModel):
    openai_key: str
    pinecone_key: Optional[str]
    pinecone_env: Optional[str]


class Api_response(BaseModel):
    response: str


class Train_request(BaseModel):
    urls: List[str]
    store: Optional[str] = "FAISS"


@app.post("/set-api-key")
async def set_api_key(api_key: Api_key):
    """Set the OpenAI and Pinecone API keys in the environment variable."""

    if api_key.openai_key is None:
        raise ValueError("You must provide your OpenAI API KEY.")
    else:
        try:
            os.environ['OPENAI_API_KEY'] = api_key.openai_key
        except Exception as e:
            raise Exception("Could not set the OpenAI API key in the environment variable because : " + str(e))

        if api_key.pinecone_key is not None and api_key.pinecone_env is None:
            raise ValueError("PINECONE_API_ENV must be provided.")
        elif api_key.pinecone_key is not None and api_key.pinecone_env is not None:
            try:
                os.environ['PINECONE_API_KEY'] = api_key.pinecone_key
                os.environ['PINECONE_API_ENV'] = api_key.pinecone_env
            except Exception as e:
                raise Exception("Could not set the Pinecone key in the environment variable because : " + str(e))

    return {"message": "API keys set successfully."}


@app.post("/train")
async def train(train_request: Train_request):
    """Train new Q/A chatbot with data from those urls"""

    try:
        model = Model(urls=train_request.urls)
        model.train(train_request.urls, train_request.store)
        return {"message": "Model trained successfully."}

    except Exception as e:
        raise Exception("Could not train the model because : " + str(e))


@app.get("/answer")
async def ask(query: str):
    """Ask a question to the chatbot"""

    try:

        response = model.answer(query)
        return {"response": response}

    except Exception as e:
        raise Exception("Could not ask the question because : " + str(e))

import uvicorn

uvicorn.run(app)