"""This module contains the IeeeAssistant class, which initializes and manages
the configuration and execution of an assistant model."""

import dataclasses
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chat_models.base import BaseChatModel
from langchain_core.documents import Document
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_google_genai import ChatGoogleGenerativeAI

from src.core.retriever import Retriever
from src.core.utils import read_yaml_file

load_dotenv()

ASSISTANT_CONFIG_PATH = "src/config/ieee_assistant.yaml"


@dataclasses.dataclass
class LlmConfig:
    """
    LlmConfig is a data class that holds the configuration for a language model

    Attributes:
        sys_prompt (str): The system prompt to be used by the language model.
        model (str): The name or identifier of the language model.
        temperature (float): The temperature parameter for the language model,
        which controls the randomness of the output.
    """

    sys_prompt: str
    model: str
    temperature: float


class IeeeAssistant:
    """
    IeeeAssistant is a class that initializes and manages the configuration and
    execution of an assistant model.

    Attributes:
        configs (List[dict[str, Any]]): List of configuration dictionaries.
        assistant_config (dict[str, Any]): Dictionary containing assistant
        configuration.
        sys_prompt (str): System message prompt template.
        llm_model (str): Language model identifier.
        llm_temperature (float): Temperature setting for the language model.
        tools (Any): Tools configuration.

    Methods:
        get_configs(config_path: str) -> List[dict[str, Any]]:
            Static method to retrieve configurations from a given path.

        documents_retriever(query: str):
            Static method to retrieve documents based on a query.

        get_assistant_config() -> List[dict[str, Any]]:
            Retrieves the assistant configuration from the loaded configs.

        get_llm_model() -> str:
            Retrieves and initializes the language model based on the
            configuration.

        get_assistant():
            Initializes the assistant with the language model and prompt
            templates.

        run_assistant(inputs: str):
            Runs the assistant with the given input and retrieves the response.
    """

    def __init__(self):
        self.configs = self.get_configs(config_path=ASSISTANT_CONFIG_PATH)
        self.assistant_config = self.get_assistant_config()
        self.llm_config = LlmConfig(
            sys_prompt=self.assistant_config.get('system_message'),
            model=self.assistant_config.get('model'),
            temperature=self.assistant_config.get('temperature'),
        )
        self.tools = self.assistant_config.get('tools')

    @staticmethod
    def get_configs(config_path: str) -> List[Dict[str, Any]]:
        """
        Retrieves configuration data from YAML files located at the specified
        path.

        Args:
            config_path (str): The path to a directory containing YAML files or
            a single YAML file.

        Returns:
            List[dict[str, Any]]: A list of dictionaries containing the
            configuration data.

        Raises:
            ValueError: If the provided config_path is neither a directory nor
            a YAML file.
        """
        configs = []
        if os.path.isdir(config_path):
            for filename in os.listdir(config_path):
                if filename.endswith('.yaml'):
                    filepath = os.path.join(config_path, filename)
                    configs.append(read_yaml_file(filepath))
        elif os.path.isfile(config_path) and config_path.endswith('.yaml'):
            configs.append(read_yaml_file(config_path))
        else:
            raise ValueError(f'Invalid config path: {config_path}')
        return configs

    @staticmethod
    def documents_retriever(query: str) -> list[Document]:
        """
        Retrieve documents based on the given query.

        Args:
            query (str): The search query to retrieve documents.

        Returns:
            list: A list of documents retrieved based on the query.
        """
        docs = Retriever().query_rag(query)
        return docs

    def get_assistant_config(self) -> List[dict[str, Any]]:
        """
        Retrieve the assistant configuration.

        Returns:
            List[dict[str, Any]]: A list of dictionaries containing the
            assistant configuration.
        """
        for cfg in self.configs:
            return cfg.get('config')

    def get_llm_model(self) -> BaseChatModel:
        """
        Returns the appropriate LLM model instance based on the configuration.

        If the model specified in the configuration is 'gemini', it returns an
        instance
        of ChatGoogleGenerativeAI with the specified model and temperature
        settings.
        Otherwise, it raises a ValueError indicating that the model is invalid
        or not supported.

        Returns:
            str: An instance of ChatGoogleGenerativeAI if the model is 'gemini'

        Raises:
            ValueError: If the model specified in the configuration is invalid
            or not supported.
        """
        if 'gemini' in self.llm_config.model:
            return ChatGoogleGenerativeAI(
                model=f'{self.llm_config.model}',
                temperature=self.llm_config.temperature,
            )
        else:
            raise ValueError(
                f'Invalid LLM model: {self.llm_config.model}, or not supported'
            )

    def get_assistant(self):
        """
        Initializes the assistant by setting up the language model (LLM) and
        creating a prompt template.

        This method performs the following steps:

        1. Retrieves the LLM model using the `get_llm_model` method and
        assigns it to `self._llm`.

        2. Constructs a list of message templates for the assistant's prompt,
        including system and human message templates.

        3. Defines the input variables required for the prompt.

        4. Creates a `ChatPromptTemplate` using the defined messages and
        input variables.

        5. Initializes the assistant by creating a document chain with the
        LLM and the prompt template.

        Returns:
            None
        """
        if not hasattr(self, 'assistant') or not hasattr(self, '_llm'):
            llm = self.get_llm_model()
            self._llm = llm

            messages = [
                SystemMessagePromptTemplate.from_template(
                    self.llm_config.sys_prompt
                ),
                HumanMessagePromptTemplate.from_template('{input}'),
                # MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
            input_variables = ['input', 'context', 'chat_history']

            _prompt = ChatPromptTemplate(
                messages=messages, input_variables=input_variables
            )
            _document_prompt = PromptTemplate.from_template(
                'Documento:{source}, pagina:{page}, conteudo: {page_content}'
            )

            self.assistant = create_stuff_documents_chain(
                self._llm, _prompt, document_prompt=_document_prompt
            )

    def run_assistant(self, inputs: str, chat_history):
        """
        Runs the assistant with the given input string.

        This method retrieves relevant documents based on the input string and
        invokes the assistant with the input and the retrieved context.

        Args:
            inputs (str): The input string to process.

        Returns:
            The response from the assistant.

        Raises:
            ValueError: If the assistant is not initialized.
        """
        if not hasattr(self, 'assistant'):
            raise ValueError('Assistant not initialized')
        docs = self.documents_retriever(inputs)
        # print('Contexto:', docs)

        response = self.assistant.stream({
            'input': inputs,
            'context': docs,
            'chat_history': chat_history,
        })

        # print(response)
        return response
