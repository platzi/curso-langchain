from typing import Dict, List

from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import CohereEmbeddings, OpenAIEmbeddings
from langchain.memory import VectorStoreRetrieverMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from rich.console import Console
from utils import (
    TransformersDocsJSONLLoader,
    get_cohere_api_key,
    get_file_path,
    get_openai_api_key,
    get_query_from_user,
    load_config,
)

console = Console()

with open("hashira/prompt_ai_conversation.txt", "r") as file:
    PROMPT = file.read()

PROMPT_TEMPLATE_CHAT = PromptTemplate(
    input_variables=["history", "input"], template=PROMPT
)


def select_embedding_provider(provider: str, model: str):
    if provider.lower() == "openai":
        get_openai_api_key()
        return OpenAIEmbeddings(model=model)
    elif provider.lower() == "cohere":
        get_cohere_api_key()
        return CohereEmbeddings(model=model)
    else:
        raise ValueError(
            f"Proveedor de embedding no compatible: {provider}. Los proveedores admitidos son 'OpenAI' y 'Cohere'."
        )


def load_documents(file_path: str) -> List[Dict]:
    config = load_config()
    loader = TransformersDocsJSONLLoader(file_path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["text_splitting"]["chunk_size"],
        chunk_overlap=config["text_splitting"]["chunk_overlap"],
        length_function=len,
    )

    return text_splitter.split_documents(data)


def get_chroma_db(embeddings, documents, path):
    config = load_config()
    if config["recreate_chroma_db"]:
        console.print("Recreando Chroma DB...")
        return Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=path,
        )
    else:
        console.print("Cargando Chroma DB existente...")
        return Chroma(persist_directory=path, embedding_function=embeddings)


def process_query(query: str, vectorstore) -> str:
    config = load_config()
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": config["document_retrieval"]["k"]}
    )
    memory = VectorStoreRetrieverMemory(retriever=retriever)

    llm = ChatOpenAI(
        model_name=config["chat_model"]["model_name"],
        temperature=config["chat_model"]["temperature"],
        max_tokens=config["chat_model"]["max_tokens"],
    )

    conversation_with_summary = ConversationChain(
        prompt=PROMPT_TEMPLATE_CHAT,
        llm=llm,
        memory=memory,
        verbose=config["conversation_chain"]["verbose"],
    )

    console.print("[yellow]La IA está pensando...[/yellow]")

    return conversation_with_summary.predict(input=query)


def run_conversation(vectorstore_chroma):
    conversation_history = []

    console.print(
        "\n[blue]IA:[/blue] Hola! Cuál es tu nombre? Qué quieres preguntarme?"
    )

    while True:
        console.print("\n[blue]You:[/blue]")
        query = get_query_from_user()

        if query.lower() == "salir":
            break

        conversation_history.append({"role": "system", "content": f"User: {query}"})

        response = process_query(query=query, vectorstore=vectorstore_chroma)

        console.print(f"[red]IA:[/red] {response}")

        conversation_history.append({"role": "system", "content": f"AI: {response}"})


def main():
    config = load_config()

    embeddings = select_embedding_provider(
        config["embeddings_provider"], config["embeddings_model"]
    )

    documents = load_documents(get_file_path())

    vectorstore_chroma = get_chroma_db(embeddings, documents, config["chroma_db_name"])

    console.print(f"[green]Documentos {len(documents)} cargados.[/green]")

    run_conversation(vectorstore_chroma)


if __name__ == "__main__":
    main()
