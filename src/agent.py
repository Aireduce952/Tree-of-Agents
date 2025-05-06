import backoff
from openai import OpenAI, RateLimitError, APIError, APIConnectionError
from requests.exceptions import JSONDecodeError, ConnectTimeout
from urllib3.exceptions import ConnectTimeoutError
from typing import Optional


class BaseAgent:
    def __init__(self, id: Optional[str] = None, api_key: Optional[str] = None,
                 name: Optional[str] = None, model: Optional[str] = None,
                 base_url: Optional[str] = None):
        """
        Base agent class for LLM interactions. Subclasses should implement generate_response.
        """
        self.id = id
        self.api_key = api_key
        self.name = name
        self.model = model
        self.base_url = base_url  # Added to support flexible deployments

        self.explanation = ""
        self.claim = ""
        self.docs = ""
        self.fact = ""
        self.conclusion = ""
        self.inspired = ""
        self.message = []
        self.helpful = {}
        self.opinions = {}
        self.sequence = []
        self.useless_sequence = []
        self.final_decision = ""
        self.last = ""
        self.options = ""

    def print_attributes(self):
        """Print all agent attributes for debugging."""
        print(f"Agent attributes for {self.__class__.__name__}:")
        for attr, value in self.__dict__.items():
            print(f"  {attr}: {value}")

    def reset(self):
        """Reset agent state between runs."""
        self.__init__(self.id, self.api_key, self.name, self.model, self.base_url)

    @backoff.on_exception(
        backoff.expo,
        (RateLimitError, APIError, APIConnectionError, ValueError,
         JSONDecodeError, ConnectTimeout, ConnectTimeoutError),
        max_tries=5
    )
    def generate_response(self, message):
        raise NotImplementedError("Subclasses must implement this method.")


class DeepSeek(BaseAgent):
    def __init__(self, **kwargs):
        """
        Initialize DeepSeek agent with optional overrides.
        """
        super().__init__(**kwargs)
        if self.base_url is None:
            self.base_url = "https://api.deepseek.com"  # Default value, can be overridden

    def generate_response(self, message):
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=message,
            stream=False,
            temperature=0,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content


class LocalLlama(BaseAgent):
    def __init__(self, **kwargs):
        """
        Initialize LocalLlama agent for local deployment use.
        """
        super().__init__(**kwargs)
        if self.base_url is None:
            self.base_url = "http://127.0.0.1:8000/v1"  # Default local URL

    def generate_response(self, message):
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=message,
            temperature=0,
            top_p=0,
            max_tokens=2048,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
