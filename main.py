import os
import pickle
import time
from ..api_keys import chatGroq_key, github_key
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from getpass import getpass
from langchain_community.document_loaders import GithubFileLoader
from langchain_chroma import Chroma
from uuid import uuid4
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

print('Enter the repo name')
repoName = input()
print('Enter the branch name')
branchName = input()
print('Enter your question')
question = input()
modelName = "llama-3.1-8b-instant"

documentList = []

def cleanData(documentList):
    for i in range(0, len(documentList)):
        newData = ' '.join(documentList[i].page_content.split())
        documentList[i].page_content = newData
    return documentList

def configureTextSplitterParams():
    t = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=10,
        length_function=len,
        is_separator_regex=False,
)
    return t

def configureLLM():
    llm = ChatGroq(
        model=modelName,
        temperature=0.34,
        max_tokens=500,
        timeout=None,
        max_retries=2,
        api_key=chatGroq_key
)
    return llm
    

def configureEmbeddingModel():
    model_kwargs = {'device': 'cpu'}
    embedding_model = HuggingFaceEmbeddings(    
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs = model_kwargs
    )
    return embedding_model

def configureVectorStore():
    vector_store = Chroma(
    collection_name="code_collection",
    embedding_function=configureEmbeddingModel(),
    persist_directory="vectorDB_Code", 
    )
    return vector_store


loader = GithubFileLoader(
    repo= repoName,
    branch= branchName,
    access_token= github_key,
    github_api_url="https://api.github.com",
    file_filter=lambda file_path: file_path.endswith(('.py', '.html', '.css', '.js', '.md')),
    # file_filter=lambda path: (
    #     # allow all directories (those without a file extension)
    #     not path or '/' not in path or not path.split('/')[-1].count('.')
    # ) or path.endswith((".py", ".md", '.js'))
)

document = loader.lazy_load()

for doc in document:
    documentList.append(doc)

newDocumentList = cleanData(documentList)

text_splitter = configureTextSplitterParams()
split_data = text_splitter.split_documents(newDocumentList)

embedding_model = configureEmbeddingModel()
vector_store = configureVectorStore()

uuids = [str(uuid4()) for _ in range(len(split_data))]
vector_store.add_documents(documents=split_data, ids=uuids)
retriver = vector_store.as_retriever()

query = question
results = vector_store.similarity_search(query, k=4)
res = []
for result in results:
    res.append(result.page_content)


prompt_template = PromptTemplate(
    input_variables=['Question', 'Data'],
    template= """ 
        Question: {Question}
        Answer the above question. Take the below context as reference. Make sure to write atleast 3 sentences.
        '{Data}'
    """
)

chain = LLMChain(llm = configureLLM(), prompt=prompt_template, verbose=True)

print("cool")

final_output = chain.invoke({
    "Question" : question,
    "Data" : res
})

print(final_output['text'])
vector_store._client.delete_collection(vector_store._collection.name)



