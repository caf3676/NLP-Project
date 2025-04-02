import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage

#Load the API key
os.environ['OPENAI_API_KEY'] = "INSERT OPENAPI KEY HERE"

#Recursively splits text into chunks
def get_text_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0, separators=[" ", ",", "\n"])
    chunks = text_splitter.split_text(raw_text)
    return chunks

#Generates a vectorstore
def get_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

#Generates the model template
def get_conversation_chain(vectorstore):
    template = ChatPromptTemplate.from_messages(
      [
        SystemMessage(
            content=(
                "You are an autoregressive language model that has been fine-"
                "tuned with instruction-tuning and RLHF. You carefully "
                "provide accurate, factual, thoughtful, nuanced answers, and"
                "are brilliant at reasoning. If you think there might not be "
                "a correct answer, you say so. Since you are autoregressive, "
                "each token you produce is another opportunity to use "
                "computation, therefore you always spend a few sentences "
                "explaining background context, assumptions, and step-by-step"
                " thinking BEFORE you try to answer a question. Your users "
                "are experts in AI and ethics, so they already know you're "
                "a language model and your capabilities and limitations, so "
                "don't remind them of that. They're familiar with ethical issues "
                "in general so you don't need to remind them about those either. "
                "Don't be verbose in your answers, but do provide details and "
                "examples where it might help the explanation."
            )
        ),
        HumanMessagePromptTemplate.from_template("{text}"),
      ]
    )
    llm = ChatOpenAI()#model_name="gpt-4")
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        condense_question_prompt=template,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

#Putting it all together
def create_chat_from_text(directory):
    with open('combined.txt', 'r') as file:
        text_documents = file.read()
    chunks = get_text_chunks(text_documents)
    vs = get_vectorstore(chunks)
    chain = get_conversation_chain(vs)
    return chain

#Testing
chat_chain = create_chat_from_text(".")
result = chat_chain.run("What is multipoint simulation? Explain in five sentences.")
print(result)
