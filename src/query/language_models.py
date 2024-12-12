import io
import json
import random
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Union

import anthropic
import dashscope
import google.generativeai as genai
import numpy as np
import openai
from mistralai import Mistral

from src.query.utils import extract_probability, retry_on_model_failure


class LanguageModel(ABC):

    def __init__(self, model_version):
        self.model_version = model_version

    @retry_on_model_failure(max_retries=3)
    def make_forecast(
        self,
        forecasting_question: str,
        context: str,
        verbose_reasoning: bool = False,
        **kwargs,
    ) -> Tuple[int, str]:
        """Make a forecast using the model.

        Parameters
        ----------
        forecasting_question : str
            Question to be forecasted.
        context : str
            System prompt and context.
        verbose_reasoning : bool, optional
            Print model output., by default False

        Returns
        -------
        Tuple[int, str]
            Tuple with forecasted answer and reasoning.
        """
        reply = self.query_model(forecasting_question, context, **kwargs)
        if verbose_reasoning:
            print("Given answer was:\n", reply)
        return (extract_probability(reply), reply)

    @abstractmethod
    def query_model(
        self,
        user_prompt: str,
        system_prompt: str,
        return_details: bool = False,
        **kwargs,
    ) -> Union[str, Any]:
        """Query the model with a user and system prompt.

        Parameters
        ----------
        user_prompt : str
            User prompt (specific task)
        system_prompt : str
            System prompt (general context and role description).
        return_details : bool, optional
            If True return all details the model's API return (attention: in
            this case the returned values have not the same format for all
            inherited classes). If False, only the response text is returned
            (i.e. same format for all inherited classed). , by default False

        Returns
        -------
        Union[str, Any]
            Model response text (if return_details==False) or detailed response object.
        """
        pass


class LanguageModelWithBatchAPI(LanguageModel):

    @abstractmethod
    def query_batch(
        self, user_prompts: List[str], system_prompts: List[str]
    ) -> (str, List[str]):
        """Query the model with a batch of user and system prompts.

        Parameters
        ----------
        user_prompts : List[str]
            List of user prompts (specific tasks).
        system_prompts : List[str]
            List of system prompts (general context and role description).

        Returns
        -------
        (str, List[str])
            batch id, which can be used to retrieve batch and list of request ids.

        """
        pass

    @abstractmethod
    def check_batch_status(self, batch_id: str) -> str:
        """Check the status of a batch.

        Parameters
        ----------
        batch_id : str

        Returns
        -------
        str
            Status of the batch.
        """
        pass

    @abstractmethod
    def cancel_batch(self, batch_id: str) -> bool:
        """Cancel a batch.

        Parameters
        ----------
        batch_id : str

        Returns
        -------
        bool
            Returns True if the batch was successfully cancelled.
        """
        pass

    @abstractmethod
    def retrieve_batch(
        self, batch_id: str, request_ids: List[str], return_details: bool = False
    ) -> List[Union[str, Any]]:
        """Retrieves the results of a batch if possible.

        Parameters
        ----------
        batch_id : str
        request_ids : List[str]
            List of request ids
        return_details : bool, optional
            If True return all details the model's API return (attention: in
            this case the returned values have not the same format for all
            inherited classes). If False, only the response text is returned
            (i.e. same format for all inherited classed). , by default False

        Returns
        -------
        List[Union[str, Any]]
            Model response texts (if return_details==False) or detailed response object.
        """
        pass


class OpenAIModel(LanguageModelWithBatchAPI):

    def __init__(self, api_key, model_version="gpt-4o-mini"):
        """Generate an OpenAI model instance.

        Parameters
        ----------
        api_key : str
        model_version : str, optional
            Available options are:
            gpt-4o", gpt-4o-mini, gpt-4-turbo, o1-preview, o1-mini, by default "gpt-4o-mini"
        """
        # can we have a bit of documentation here that says which model_versions
        # there are (only the most important ones for us) and maybe a quick pro
        # / con if that's not self evident from the model strings
        super().__init__(model_version)
        self.client = openai.OpenAI(
            api_key=api_key,
        )

    def query_model(
        self, user_prompt: str, system_prompt: str, return_details=False, **kwargs
    ) -> Union[str, Any]:
        assert not any(
            key in kwargs for key in ["model", "messages", "logprobs"]
        ), "Invalid keyword argument"
        kwargs.setdefault("top_logprobs", 20)
        kwargs.setdefault("max_tokens", 600)
        response = self.client.chat.completions.create(
            model=self.model_version,
            messages=[
                {"role": "user", "content": user_prompt},
                {"role": "system", "content": system_prompt},
            ],
            logprobs=True,
            **kwargs,
        )
        if return_details:
            return response
        else:
            return response.choices[0].message.content

    def query_batch(self, user_prompts, system_prompts, **kwargs):
        assert not any(
            key in kwargs for key in ["model", "messages"]
        ), "Invalid keyword argument"
        batch_requests = []
        kwargs.setdefault("max_tokens", 600)
        request_ids = []
        for i, (user_prompt, system_prompt) in enumerate(
            zip(user_prompts, system_prompts)
        ):
            request_id = f"request-{i}"
            request_ids.append(request_id)
            request = {
                "custom_id": request_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model_version,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    **kwargs,
                },
            }
            batch_requests.append(request)

        file_like = io.BytesIO()
        for request in batch_requests:
            file_like.write((json.dumps(request) + "\n").encode("utf-8"))
        file_like.seek(0)
        batch_input_file = self.client.files.create(file=file_like, purpose="batch")
        response = self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        return (response.id, request_ids)

    def check_batch_status(self, batch_id):
        response = self.client.batches.retrieve(batch_id)
        return response.status

    def cancel_batch(self, batch_id):
        response = self.client.batches.cancel(batch_id)
        return response.status == "cancelled" or response.status == "cancelling"

    def retrieve_batch(self, batch_id, request_ids, return_details=False):
        status = self.check_batch_status(batch_id)
        if status == "completed":
            out_file = self.client.batches.retrieve(batch_id).output_file_id
            response = self.client.files.content(out_file)
            file_like = io.BytesIO()
            file_like.write(response.content)
            file_like.seek(0)
            results = []
            with file_like as file:
                for line in file:
                    # Parsing the JSON string into a dict and appending to the list of results
                    json_object = json.loads(line.strip())
                    results.append(json_object)
            if return_details:
                return {
                    result["custom_id"]: result["response"]["body"]
                    for result in results
                }
            else:
                return {
                    result["custom_id"]: result["response"]["body"]["choices"][0][
                        "message"
                    ]["content"]
                    for result in results
                }
        else:
            raise Exception(f"Batch {batch_id} is not completed. Status is: {status}")


class AnthropicModel(LanguageModel):

    def __init__(self, api_key, model_version="claude-3-5-haiku-20241022"):
        """Generate an Anthropic model instance.

        Parameters
        ----------
        api_key : str
        model_version : str, optional
            Available options are:
            claude-3-5-sonnet-20241022, claude-3-5-haiku-20241022, claude-3-opus-20240229, by default "claude-3-haiku-20240307"
        """
        super().__init__(model_version)
        self.client = anthropic.Anthropic(api_key=api_key)

    def query_model(
        self, user_prompt, system_prompt, return_details=False, **kwargs
    ) -> Union[str, Any]:
        assert not any(
            key in kwargs for key in ["model", "system", "messages"]
        ), "Invalid keyword argument"
        kwargs.setdefault("max_tokens", 600)
        response = self.client.messages.create(
            model=self.model_version,
            system=system_prompt,
            messages=[
                {"role": "user", "content": f"{user_prompt}"},
            ],
            **kwargs,
        )
        if return_details:
            return response
        else:
            return response.content[0].text


class GeminiModel(LanguageModel):

    def __init__(self, api_key, model_version="gemini-1.5-flash-8b"):
        """Generate a Gemini model instance.

        Parameters
        ----------
        api_key : str
        model_version : str, optional
            Available options are:
            gemini-1.5-flash, gemini-1.5-flash-8b,
            gemini-1.5-pro, by default "gemini-1.5-flash-8b"
        """
        super().__init__(model_version)
        genai.configure(api_key=api_key)

    def query_model(
        self, user_prompt, system_prompt, return_details=False, **kwargs
    ) -> Union[str, Any]:
        assert not any(
            key in kwargs for key in ["response_logprobs"]
        ), "Invalid keyword argument"
        kwargs.setdefault("max_output_tokens", 600)
        # kwargs.setdefault("logprobs", 5)
        config = genai.GenerationConfig(response_logprobs=False, **kwargs)
        model = genai.GenerativeModel(
            model_name=self.model_version,
            generation_config=config,
            system_instruction=system_prompt,
        )
        response = model.generate_content(user_prompt)
        if return_details:
            return response
        else:
            return response.text


class XAIModel(LanguageModel):

    def __init__(self, api_key, model_version="grok-beta"):
        """Generate a XAI model instance.

        Parameters
        ----------
        api_key : str
        model_version : str, optional
            Available options are: grok-beta, by default "grok-beta"
        """
        super().__init__(model_version)
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
        )

    def query_model(
        self, user_prompt, system_prompt, return_details=False, **kwargs
    ) -> Union[str, Any]:
        assert not any(
            key in kwargs for key in ["model", "messages", "logprobs"]
        ), "Invalid keyword argument"
        kwargs.setdefault("max_tokens", 600)
        response = self.client.chat.completions.create(
            model=self.model_version,
            messages=[
                {"role": "user", "content": f"{user_prompt}"},
                {"role": "system", "content": f"{system_prompt}"},
            ],
            **kwargs,
        )
        if return_details:
            return response
        else:
            return response.choices[0].message.content


class LLAMAModel(LanguageModel):

    def __init__(self, api_key, model_version="llama3.1-8b"):
        """Generate a Llama model instance.

        Parameters
        ----------
        api_key : str
        model_version : str, optional
            Available options are: llama3.2-1b, llama3.2-3b,
            llama3.2-11b-vision, llama3.2-90b-vision, llama3.1-8b, llama3.1-70b,
            llama3.1-405b, by default "llama3.1-8b"
        """
        super().__init__(model_version)
        self.client = openai.OpenAI(
            api_key=api_key, base_url="https://api.llama-api.com"
        )

    def query_model(
        self, user_prompt, system_prompt, return_details=False, **kwargs
    ) -> Union[str, Any]:
        assert not any(
            key in kwargs for key in ["model", "messages", "logprobs"]
        ), "Invalid keyword argument"
        kwargs.setdefault("top_logprobs", 20)
        kwargs.setdefault("max_tokens", 600)
        response = self.client.chat.completions.create(
            model=self.model_version,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": user_prompt},
            ],
            logprobs=True,
            **kwargs,
        )
        if return_details:
            return response
        else:
            return response.choices[0].message.content


class MistralModel(LanguageModel):

    def __init__(self, api_key, model_version="mistral-small-2409"):
        """Generate a Mistral model instance.

        Parameters
        ----------
        api_key : str
        model_version : str, optional
            Available options are: mistral-large-2407, mistral-small-2409 ,
            mistral-small-2402 , ministral-8b-2410, ministral-3b-2410, by default "mistral-small-2409"
        """
        super().__init__(model_version)
        self.client = Mistral(api_key=api_key)

    def query_model(
        self, user_prompt, system_prompt, return_details=False, **kwargs
    ) -> Union[str, Any]:
        assert not any(
            key in kwargs for key in ["model", "messages", "logprobs"]
        ), "Invalid keyword argument"
        kwargs.setdefault("max_tokens", 600)
        response = self.client.chat.complete(
            model=self.model_version,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": user_prompt},
            ],
            **kwargs,
        )
        if return_details:
            return response
        else:
            return response.choices[0].message.content


class QwenModel(LanguageModel):

    def __init__(self, api_key, model_version="qwen-turbo"):
        """Generate a Qwen model instance.

        Parameters
        ----------
        api_key : str
        model_version : str, optional
            Available options are: qwen-max, qwen-plus ,
            qwen-turbo, by default "qwen-turbo"
        """
        super().__init__(model_version)
        dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"
        self.api_key = api_key

    def query_model(
        self, user_prompt, system_prompt, return_details=False, **kwargs
    ) -> Union[str, Any]:
        assert not any(
            key in kwargs for key in ["model", "messages"]
        ), "Invalid keyword argument"
        kwargs.setdefault("max_tokens", 600)
        kwargs.setdefault("seed", random.randint(1, 10000))
        response = dashscope.Generation.call(
            model=self.model_version,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            api_key=self.api_key,
            **kwargs,
        )
        if return_details:
            return response
        else:
            if response.output is None:
                raise Exception(f"response.output is None. response is: {response}")
            return response.output.text
