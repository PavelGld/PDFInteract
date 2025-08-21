import functools
import inspect
from typing import Callable, cast, Literal, List, Union, Optional, Dict
import httpx
import pydantic
from openai import OpenAI, Stream, APIResponseValidationError, AsyncOpenAI
from openai._base_client import make_request_options
from openai._models import validate_type, construct_type, BaseModel
#from pydantic import BaseModel
from openai._resource import SyncAPIResource
from openai._types import ResponseT, ModelBuilderProtocol, NotGiven, NOT_GIVEN, Headers, Query, Body
from openai._utils import maybe_transform, required_args
from openai.resources.chat import Completions as ChatCompletions
from openai.resources import Completions
from openai.types import CreateEmbeddingResponse, Completion, Embedding
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam, completion_create_params, \
    ChatCompletionToolChoiceOptionParam, ChatCompletionToolParam, ChatCompletionChunk

from langchain_openai import ChatOpenAI as GPT
from langchain_openai import OpenAIEmbeddings as OpenAIEmbeds
from langchain_core.utils import convert_to_secret_str


class ChatGPTEntry(BaseModel):
    role: str
    content: str


class ResponseSchema(BaseModel):
    response: ChatGPTEntry
    prompt_tokens: int
    completion_tokens: int
    available_tokens: int
    raw_openai_response: Union[ChatCompletion, Completion, None]  = None


def chat_completion_overload(func: Callable):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # trick to get openai schema from NDT custom schema
        # here is not ChatCompletion, here NDT schema but wrong casted ChatCompletion to inside openai lib
        #print(args, kwargs)
        result: ChatCompletion | Stream = func(*args, **kwargs)
        if isinstance(result, Stream):
            return result

        #print(result)
        ndt_response = ResponseSchema(**result.model_dump(exclude_unset=True, exclude_defaults=True))
        #print(ndt_response.available_tokens)
        return ndt_response.raw_openai_response

    return wrapper


class NDTChatCompletions(ChatCompletions):
    
    @required_args(["messages", "model"], ["messages", "model", "stream"])
    def create(
            self,
            *,
            messages: List[ChatCompletionMessageParam],
            model: Union[
                str,
                Literal[
                    "gpt-4-1106-preview",
                    "gpt-4-vision-preview",
                    "gpt-4",
                    "gpt-4-0314",
                    "gpt-4-0613",
                    "gpt-4-32k",
                    "gpt-4-32k-0314",
                    "gpt-4-32k-0613",
                    "gpt-3.5-turbo-1106",
                    "gpt-3.5-turbo",
                    "gpt-3.5-turbo-16k",
                    "gpt-3.5-turbo-0301",
                    "gpt-3.5-turbo-0613",
                    "gpt-3.5-turbo-16k-0613",
                ],
            ],
            frequency_penalty: Optional[float] = NOT_GIVEN,
            function_call: completion_create_params.FunctionCall = NOT_GIVEN,
            functions: List[completion_create_params.Function] = NOT_GIVEN,
            logit_bias: Optional[Dict[str, int]] = NOT_GIVEN,
            max_tokens: Optional[int] = NOT_GIVEN,
            n: Optional[int] = NOT_GIVEN,
            presence_penalty: Optional[float] = NOT_GIVEN,
            response_format: completion_create_params.ResponseFormat = NOT_GIVEN,
            seed: Optional[int] = NOT_GIVEN,
            stop: Union[Optional[str], List[str]] = NOT_GIVEN,
            stream: Optional[Literal[False]] = NOT_GIVEN,
            temperature: Optional[float] = NOT_GIVEN,
            tool_choice: ChatCompletionToolChoiceOptionParam = NOT_GIVEN,
            tools: List[ChatCompletionToolParam] = NOT_GIVEN,
            top_p: Optional[float] = NOT_GIVEN,
            user: str = NOT_GIVEN,
            # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
            # The extra values given here take precedence over values defined on the client or passed to this method.
            extra_headers: Headers = None,
            extra_query: Query = None,
            extra_body: Body = None,
            timeout: float = NOT_GIVEN,
    ) -> ChatCompletion:
        result: ResponseSchema = self._post(
            "/chat/completions",
            body=maybe_transform(
                {
                    "messages": messages,
                    "model": model,
                    "frequency_penalty": frequency_penalty,
                    "function_call": function_call,
                    "functions": functions,
                    "logit_bias": logit_bias,
                    "max_tokens": max_tokens,
                    "n": n,
                    "presence_penalty": presence_penalty,
                    "response_format": response_format,
                    "seed": seed,
                    "stop": stop,
                    "stream": stream,
                    "temperature": temperature,
                    "tool_choice": tool_choice,
                    "tools": tools,
                    "top_p": top_p,
                    "user": user,
                },
                completion_create_params.CompletionCreateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=ResponseSchema,
            stream=stream or False,
            stream_cls=Stream[ChatCompletionChunk],
        )

        #print(result)
        return result.raw_openai_response


class NDTCompletions(Completions):
    
    @required_args(["model", "prompt"], ["model", "prompt", "stream"])
    def create(
        self,
        *,
        model: Union[
            str,
            Literal[
                "babbage-002",
                "davinci-002",
                "gpt-3.5-turbo-instruct",
                "text-davinci-003",
                "text-davinci-002",
                "text-davinci-001",
                "code-davinci-002",
                "text-curie-001",
                "text-babbage-001",
                "text-ada-001",
            ],
        ],
        prompt: Union[str, List[str], List[int], List[List[int]], None],
        best_of: Optional[int] = NOT_GIVEN,
        echo: Optional[bool] = NOT_GIVEN,
        frequency_penalty: Optional[float] = NOT_GIVEN,
        logit_bias: Optional[Dict[str, int]] = NOT_GIVEN,
        logprobs: Optional[int] = NOT_GIVEN,
        max_tokens: Optional[int] = NOT_GIVEN,
        n: Optional[int] = NOT_GIVEN,
        presence_penalty: Optional[float] = NOT_GIVEN,
        seed: Optional[int] = NOT_GIVEN,
        stop: Union[Optional[str], List[str], None] = NOT_GIVEN,
        stream: Optional[Literal[False]] = NOT_GIVEN,
        suffix: Optional[str] = NOT_GIVEN,
        temperature: Optional[float] = NOT_GIVEN,
        top_p: Optional[float] = NOT_GIVEN,
        user: str = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers = None,
        extra_query: Query = None,
        extra_body: Body = None,
        timeout: float = NOT_GIVEN,
    ) -> Completion:
        result: ResponseSchema = self._post(
            "/completions",
            body=maybe_transform(
                {
                    "model": model,
                    "prompt": prompt,
                    "best_of": best_of,
                    "echo": echo,
                    "frequency_penalty": frequency_penalty,
                    "logit_bias": logit_bias,
                    "logprobs": logprobs,
                    "max_tokens": max_tokens,
                    "n": n,
                    "presence_penalty": presence_penalty,
                    "seed": seed,
                    "stop": stop,
                    "stream": stream,
                    "suffix": suffix,
                    "temperature": temperature,
                    "top_p": top_p,
                    "user": user,
                },
                completion_create_params.CompletionCreateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=ResponseSchema,
            stream=stream or False,
            stream_cls=Stream[Completion],
        )
        
        #print(result)
        import time
        time.sleep(5)
        return result.raw_openai_response


    
class NDTChat(SyncAPIResource):
    completions: NDTChatCompletions

    def __init__(self, client: OpenAI) -> None:
        super().__init__(client)
        self.completions = NDTChatCompletions(client)


class EmbeddingResponseSchema(BaseModel):
    data: list[Embedding]
    prompt_tokens: int
    available_tokens: int
    raw_openai_response: CreateEmbeddingResponse = None


def embeddings_overload(func: Callable):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # trick to get openai schema from NDT custom schema
        # here is not CreateEmbeddingResponse, here NDT schema but wrong casted CreateEmbeddingResponse to inside openai lib
        result: CreateEmbeddingResponse = func(*args, **kwargs)
        ndt_response = EmbeddingResponseSchema(**result.model_dump(exclude_unset=True, exclude_defaults=True))
        #print(ndt_response.available_tokens)
        return ndt_response.raw_openai_response

    return wrapper


class NDTOpenAI(OpenAI):
    # chat: NDTChat
    # completions: NDTCompletions
    server_url: str = "https://api.neuraldeep.tech/"

    def __init__(self, api_key, **kwargs):
        super().__init__(api_key=api_key, base_url=self.server_url, **kwargs)
        # self.embeddings.create = embeddings_overload(self.embeddings.create)
        # self.chat = NDTChat(self)
        # self.completions = NDTCompletions(self)

class AsyncNDTOpenAI(AsyncOpenAI):
    server_url: str = "https://api.neuraldeep.tech/"
    def __init__(self, api_key, **kwargs):
        super().__init__(api_key=api_key, base_url=self.server_url, **kwargs)
        
        
class ChatOpenAI(GPT):
        
    '''
    Класс ChatOpenAI по аналогии с одноименным классом из библиотеки langchain
    '''
    
    openai_api_key: str = convert_to_secret_str('api_key')
    
    def __init__(self, course_api_key, **kwargs):
        super().__init__(client = NDTOpenAI(api_key=course_api_key).chat.completions, async_client= AsyncNDTOpenAI(api_key=course_api_key).chat.completions, **kwargs)


class OpenAIEmbeddings(OpenAIEmbeds):
    
    '''
    Класс OpenAIEmbeddings по аналогии с одноименным классом из библиотеки langchain
    '''
    
    openai_api_key: str = convert_to_secret_str('api_key')
    
    
    def __init__(self, course_api_key, **kwargs):
        super().__init__(client = NDTOpenAI(api_key=course_api_key).embeddings, async_client= AsyncNDTOpenAI(api_key=course_api_key).embeddings, **kwargs)


class AiTunnelEmbeddings:
    """
    Класс для работы с AiTunnel API эмбеддингами
    """
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-large"):
        """
        Инициализация клиента AiTunnel
        
        Args:
            api_key: AiTunnel API ключ (sk-aitunnel-xxx)
            model: Модель для эмбеддингов
        """
        if not api_key or not api_key.startswith("sk-aitunnel-"):
            raise ValueError(f"Invalid AiTunnel API key format. Expected: sk-aitunnel-xxx, got: {api_key[:20]}...")
            
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.aitunnel.ru/v1/"
        )
        self.model = model
        print(f"Initialized AiTunnel client with model: {model}")
        
    def test_connection(self):
        """Тест подключения к AiTunnel API"""
        try:
            test_response = self.client.embeddings.create(
                input="test connection",
                model=self.model
            )
            print("✓ AiTunnel API connection successful")
            return True
        except Exception as e:
            print(f"✗ AiTunnel API connection failed: {e}")
            return False
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Создать эмбеддинги для списка текстов с учетом rate limiting
        
        Args:
            texts: Список текстов для векторизации
            
        Returns:
            Список векторных представлений
        """
        import time
        
        # Обрабатываем тексты по одному с большими паузами для гарантированного избежания rate limit
        all_embeddings = []
        
        print(f"Processing {len(texts)} texts with AiTunnel API (safe mode - 2 second delays)...")
        
        for i, text in enumerate(texts):
            print(f"Processing text {i + 1}/{len(texts)} (remaining: {len(texts) - i - 1})")
            
            try:
                embeddings_response = self.client.embeddings.create(
                    input=text,  # Всегда отправляем один текст
                    model=self.model
                )
                
                # Извлекаем вектор из ответа
                embedding = embeddings_response.data[0].embedding
                all_embeddings.append(embedding)
                
                # Большая пауза между запросами (30 запросов в минуту максимум)
                if i + 1 < len(texts):
                    print(f"Waiting 2 seconds before next request...")
                    time.sleep(2.0)  # 2 секунды между запросами - очень консервативно
                    
            except Exception as e:
                print(f"Ошибка при создании эмбеддингов для батча {i//batch_size + 1}: {e}")
                print(f"Error type: {type(e).__name__}")
                print(f"Using model: {self.model}")
                
                # Если 403 ошибка - увеличиваем паузу и повторяем
                if "403" in str(e) or "rate" in str(e).lower():
                    print("Rate limit или authentication error detected, увеличиваем паузу...")
                    time.sleep(3.0)
                    try:
                        input_data = batch[0] if len(batch) == 1 else batch
                        embeddings_response = self.client.embeddings.create(
                            input=input_data,
                            model=self.model
                        )
                        if len(batch) == 1:
                            batch_embeddings = [embeddings_response.data[0].embedding]
                        else:
                            batch_embeddings = [data.embedding for data in embeddings_response.data]
                        all_embeddings.extend(batch_embeddings)
                        print(f"Retry успешен для батча {i//batch_size + 1}")
                    except Exception as retry_e:
                        print(f"Повторная ошибка для батча {i//batch_size + 1}: {retry_e}")
                        raise retry_e
                else:
                    print(f"Неожиданная ошибка: {e}")
                    raise e
        
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        Создать эмбеддинг для одного запроса с обработкой rate limiting
        
        Args:
            text: Текст запроса
            
        Returns:
            Векторное представление
        """
        import time
        
        try:
            embeddings_response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            
            return embeddings_response.data[0].embedding
            
        except Exception as e:
            # Если 403 ошибка - делаем паузу и повторяем
            if "403" in str(e):
                print("Rate limit exceeded для запроса, повторяем через 2 секунды...")
                time.sleep(2.0)
                try:
                    embeddings_response = self.client.embeddings.create(
                        input=text,
                        model=self.model
                    )
                    return embeddings_response.data[0].embedding
                except Exception as retry_e:
                    print(f"Повторная ошибка при создании эмбеддинга запроса: {retry_e}")
                    raise retry_e
            else:
                print(f"Ошибка при создании эмбеддинга запроса через AiTunnel: {e}")
                raise e
