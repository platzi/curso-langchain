from typing import Dict, List

from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import CohereEmbeddings, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from rich.console import Console
from utils import (DocsJSONLLoader, get_cohere_api_key, get_file_path,
                   get_openai_api_key, get_query_from_user, load_config)

console = Console()


def load_documents(file_path: str) -> List[Dict]:
    """
    Carga los documentos desde un archivo JSONL y los divide en trozos.

    Args:
        file_path (str): Ruta al archivo JSONL.

    Returns:
        Los documentos cargados y divididos en trozos.
    """
    config = load_config()
    loader = DocsJSONLLoader(file_path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["text_splitting"]["chunk_size"],
        chunk_overlap=config["text_splitting"]["chunk_overlap"],
        length_function=len,
    )

    return text_splitter.split_documents(data)


# TODO: Implementar uso de embeddings de Cohere.
def select_embedding_provider(provider: str, model: str):
    """
    Selecciona el proveedor de embeddings para el chatbot.

    Args:
        provider (str): El proveedor de embeddings. 'openai' o 'cohere'.
        model (str): El modelo a usar para los embeddings.

    Returns:
        El objeto embeddings del proveedor seleccionado.
    """
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


def get_chroma_db(embeddings, documents, path):
    """
    Obtiene la base de datos Chroma. Crea una nueva o carga una existente.

    Args:
        embeddings: El objeto embeddings a usar.
        documents: Los documentos a indexar en la base de datos.
        path: La ruta a la base de datos Chroma.

    Returns:
        El objeto de la base de datos Chroma.
    """
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


def process_qa_query(query: str, retriever, llm) -> str:
    """
    Procesa una consulta del usuario y genera una respuesta del chatbot.

    Args:
        query (str): La consulta del usuario.
        retriever: El objeto encargado de recuperar los documentos.
        llm: El modelo de lenguaje de gran tamaño a usar.

    Returns:
        La respuesta generada por el chatbot.
    """
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever
    )
    console.print("[yellow]La IA está pensando...[/yellow]")
    return qa_chain.run(query)


def process_conversation_query(
    query: str, retriever, llm: ChatOpenAI, chat_history: List
) -> str:
    """
    Procesa una consulta del usuario y genera una respuesta del chatbot en modo de conversación.

    Args:
        query (str): La consulta del usuario.
        retriever: El objeto de recuperación de datos donde buscar la respuesta.
        llm: El modelo de lenguaje de gran tamaño a usar.
        chat_history: The list of previous question-answer pairs in the conversation.

    Returns:
        La respuesta generada por el chatbot.
    """
    config = load_config()

    conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        verbose=config["conversation_chain"]["verbose"],
    )

    console.print("[yellow]La IA está pensando...[/yellow]")
    # print(f"La historia antes de este query: {chat_history}")
    result = conversation({"question": query, "chat_history": chat_history})
    chat_history.append((query, result["answer"]))

    return result["answer"]


def run_conversation(vectorstore, chat_type, llm):
    """
    Inicia una conversación con el usuario.

    Args:
        vectorstore: La base de datos de vectores donde buscar las respuestas.
        chat_type: El tipo de chat a usar.
    """
    config = load_config()

    console.print(
        "\n[blue]IA:[/blue] Hola 🚀! Qué quieres preguntarme sobre Transformers e inteligencia artificial en general?"
    )

    chat_history = []

    if chat_type == "memory_chat":
        console.print(
            "\n[green]Estás utilizando el chatbot en modo de conversación. Recuerda que este chatbot puede recordar partes de la conversación para generar respuestas más contextualizadas.[/green]"
        )
    elif chat_type == "qa_chat":
        console.print(
            "\n[green]Estás utilizando el chatbot en modo de preguntas y respuestas. Este chatbot genera respuestas basándose puramente en la consulta actual sin considerar el historial de la conversación.[/green]"
        )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": config["document_retrieval"]["k"]}
    )

    while True:
        console.print("\n[blue]Tú:[/blue]")
        query = get_query_from_user()

        if query.lower() == "salir":
            break

        if chat_type == "memory_chat":
            response = process_conversation_query(
                query=query, retriever=retriever, llm=llm, chat_history=chat_history
            )
        elif chat_type == "qa_chat":
            response = process_qa_query(query=query, retriever=retriever, llm=llm)

        console.print(f"[red]IA:[/red] {response}")


def main():
    """
    Función principal que se ejecuta cuando se inicia el script.
    """
    config = load_config()

    embeddings = select_embedding_provider(
        config["embeddings_provider"], config["embeddings_model"]
    )

    documents = load_documents(get_file_path())

    vectorstore_chroma = get_chroma_db(embeddings, documents, config["chroma_db_name"])

    console.print(f"[green]Documentos {len(documents)} cargados.[/green]")

    llm = ChatOpenAI(
        model_name=config["chat_model"]["model_name"],
        temperature=config["chat_model"]["temperature"],
        max_tokens=config["chat_model"]["max_tokens"],
    )

    run_conversation(vectorstore_chroma, config["chat_type"], llm)


if __name__ == "__main__":
    main()
