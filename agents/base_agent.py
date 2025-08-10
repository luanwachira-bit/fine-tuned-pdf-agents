from abc import ABC, abstractmethod
import fitz  # PyMuPDF
from typing import List

class BaseAgent(ABC):
    """
    Abstract Base Class for a document-reading AI agent.
    It defines the core functionalities that any specialized agent must implement.
    """
    def __init__(self, model_name: str):
        """
        Initializes the agent with a specific model.

        Args:
            model_name (str): The identifier for the Hugging Face model to be used.
        """
        self.model = None
        self.tokenizer = None
        self.model_name = model_name
        print(f"Initializing agent with base model: {self.model_name}")
        self._load_model()

    @abstractmethod
    def _load_model(self):
        """
        Loads the tokenizer and the model from Hugging Face.
        This is where quantization or specific device placement would happen.
        """
        pass

    def load_document(self, pdf_path: str) -> str:
        """
        Reads and extracts text from a PDF document.

        Args:
            pdf_path (str): The file path to the PDF.

        Returns:
            str: The extracted text content of the PDF.
        """
        print(f"Loading document: {pdf_path}")
        text = ""
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text += page.get_text()
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
            return ""
        
        print("Document loaded successfully.")
        return text

    @abstractmethod
    def query(self, context: str, question: str) -> str:
        """
        Answers a question based on the provided context from a document.

        Args:
            context (str): The text extracted from the PDF.
            question (str): The user's question.

        Returns:
            str: The model's generated answer.
        """
        pass

    @abstractmethod
    def fine_tune(self, training_data_path: str):
        """
        Placeholder for fine-tuning the model on specific data.

        Args:
            training_data_path (str): Path to the dataset for fine-tuning.
        """
        print(f"Fine-tuning is not yet implemented for {self.__class__.__name__}.")
        pass