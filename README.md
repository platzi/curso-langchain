# HASHIRA: Sensei virtual de documentación

## **Descripción**

El objetivo principal de este proyecto es facilitar el acceso a la información documentada en varios repositorios de GitHub, a través de un chatbot interactivo. Los usuarios pueden hacer preguntas al chatbot, que buscará la respuesta en la documentación extraída.

El nombre HASHIRA se inspira en la serie de anime "Demon Slayer". Los Hashiras son los espadachines más fuertes y experimentados del Cuerpo de Asesinos de Demonios, cada uno especializado en un estilo de lucha. De manera similar, este proyecto aspira a ser un sistema de alto rendimiento y especializado, que proporcione respuestas precisas basadas en la documentación existente.

## **Dependencias**

Este proyecto requiere las siguientes dependencias:

```
- python = "^3.11"
- requests = "^2.30.0"
- pyyaml = "^6.0"
- termcolor = "^2.3.0"
- emoji = "^2.5.0"
- transformers = "^4.30.1"
- torch = "^2.0.1"
- langchain = "^0.0.209"
- jsonlines = "^3.1.0"
- tiktoken = "^0.4.0"
- openai = "^0.27.8"
- cohere = "^4.11.2"
- chromadb = "^0.3.26"
- rich = "^13.4.2"
```

## **Instalación**

Puede instalar las dependencias y crear un entorno virtual utilizando Poetry con el siguiente comando:

``` shell
poetry install
```

Luego, puede ejecutar el proyecto con:

``` shell
poetry run python hashira/ai_conversation.py
```

## **Configuración**
El funcionamiento del proyecto se puede manipular mediante el archivo **`config.yaml`**.

## **Contribuciones**

Este proyecto es de código abierto, y apreciamos cualquier contribución. Estamos particularmente interesados en las siguientes mejoras:

- La capacidad de ejecutar el proyecto con otros Modelos de Lenguaje de Máquina (LLMs).
- Conectividad con otras bases de datos vectoriales como Faiss o Weaviate.
- Adición de una interfaz de usuario y alojamiento en Hugging Face Hub Spaces.
- La capacidad de trabajar con todo el código abierto, incluyendo LLMs de Hugging Face Hub.

No hay pautas específicas para contribuir, sólo te pedimos que nos ayudes a hacer de este proyecto algo más útil y eficiente. ¡Gracias por tu apoyo!