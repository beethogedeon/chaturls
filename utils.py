from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.vectorstores import Pinecone
from langchain.schema import Document
import pinecone
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from langchain import HuggingFacePipeline
import sys
import os
import torch
import nltk
from typing import List, Union, Optional


def set_api_key(openai_key: str, pinecone_key: Optional[str], pinecone_env: Optional[str]) -> None:
    """Set the OpenAI and Pinecone API keys in the environment variable."""
    try:
        os.environ['OPENAPI_KEY'] = openai_key
    except Exception as e:
        raise Exception("Could not set the OpenAPI key in the environment variable because : " + str(e))

    if pinecone_key is not None and pinecone_env is None:
        raise ValueError("PINECONE_API_ENV must be provided.")
    elif pinecone_key is not None and pinecone_env is not None:
        try:
            os.environ['PINECONE_API_KEY'] = pinecone_key
            os.environ['PINECONE_API_ENV'] = pinecone_env
        except Exception as e:
            print("Could not set the Pinecone key in the environment variable because : " + str(e))


def extract_data_from_urls(urls: List[str]) -> List[str]:
    """Extract the data from a list of URLs."""
    try:
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        loaders = UnstructuredURLLoader(urls=urls)
        data = loaders.load()
        return data
    except Exception as e:
        print("Could not load the data from the URLs because : " + str(e))


def split_data(data: List, separator="\n", chunk_size=1024, chunk_overlap=256) -> List[Document]:
    """Split the data into sentences."""
    try:
        text_splitter = CharacterTextSplitter(separator=separator,
                                              chunk_size=chunk_size,
                                              chunk_overlap=chunk_overlap)

        text_chunks = text_splitter.split_documents(data)

        return text_chunks
    except Exception as e:
        print("Could not split the data into sentences because : " + str(e))


def saving_in_vectorstore(data: List[Document], index_name: Optional[str], store="FAISS", embeddings_type="HF") -> Union[FAISS, Pinecone]:
    """Saving the data in the vectorstore."""
    try:
        if embeddings_type == "OPENAI":
            embeddings = OpenAIEmbeddings()
        else:
            embeddings = HuggingFaceEmbeddings()

        if store == "FAISS":
            vectorstore = FAISS.from_documents(data, embedding=embeddings)

            return vectorstore
        elif store == "PINECONE":
            pinecone_api_key = os.environ.get('PINECONE_API_KEY')
            pinecone_api_env = os.environ.get('PINECONE_API_ENV')

            try:
                pinecone.init(api_key=pinecone_api_key, environment=pinecone_api_env)
            except Exception as e:
                print("Could not init Pinecone because : " + str(e))

            try:
                # vectorstore = Pinecone.from_texts([t.page_content for t in data], embeddings, index_name=index_name)
                vectorstore = Pinecone.from_documents(data, embeddings, index_name=index_name)

                return vectorstore
            except Exception as e:
                print("Could not create the vectorstore because : " + str(e))
        else:
            raise Exception("The vectorstore is not valid.")

    except Exception as e:
        print("Could not save the data in the vectorstore because : " + str(e))


def creating_llm(pretrained_model="meta-llama/Llama-2-7b-chat-hf") -> pipeline:
    """Creating the pipeline for the LLM."""
    try:

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model, use_auth_token=True)

        model = AutoModelForCausalLM.from_pretrained(pretrained_model,
                                                     device_map='auto',
                                                     torch_dtype=torch.float16,
                                                     use_auth_token=True,
                                                     load_in_8bit=True
                                                     )

        pipe = pipeline("text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        max_new_tokens=512,
                        do_sample=True,
                        top_k=30,
                        num_return_sequences=1,
                        eos_token_id=tokenizer.eos_token_id
                        )

        llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature': 0})

        return llm
    except Exception as e:
        print("Error while creating LLM : " + str(e))


def retrieval(llm: pipeline, vectorstore: Union[FAISS, Pinecone]):
    """Initialize the Retrieval QA with Source Chain"""

    try:
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        torch.save(chain, "chatbot_chain.pt")
        return chain
    except Exception as e:
        print("Error while creating the chain : " + str(e))


def answering(chain: RetrievalQAWithSourcesChain, question: str) -> str:
    """Answering the question."""

    try:
        result = chain({"question": question}, return_only_outputs=True)
        result = result["answer"]

        return result
    except Exception as e:
        print("Error while answering the question : " + str(e))


def train_model(source: List[str], store="FAISS"):
    """Train new Q/A chatbot with data from those urls"""

    if os.environ['OPENAPI_KEY'] is None:
        raise ValueError("OPENAPI_KEY must be provided.")
    elif store != "FAISS" and os.environ['PINECONE_API_KEY'] is None:
        raise ValueError("PINECONE_API_KEY must be provided.")
    else:
        try:
            data = extract_data_from_urls(source)
            data = split_data(data)
            if store == "FAISS":
                vectorstore = saving_in_vectorstore(data, index_name=None, store=store)
            else:
                vectorstore = saving_in_vectorstore(data, index_name="chatbot", store=store)

            llm = creating_llm()
            chain = retrieval(llm, vectorstore)

            return chain
        except Exception as e:
            print("Error while training the model : " + str(e))
