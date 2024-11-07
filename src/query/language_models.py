import random
from abc import ABC, abstractmethod
from typing import Any, Tuple, Union

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
    @retry_on_model_failure(max_retries=3)
    def make_forecast_with_probs(
        self,
        forecasting_question: str,
        context: str,
        verbose_reasoning: bool = False,
        **kwargs,
    ) -> tuple:
        """Make a forecast using the model and return multiple possible
        predictions with their probabilities.

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
        tuple(List, List)
            (List of tokens, List of probabilities)
        """
        pass

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
            inherited classes). If True, only the response text is returned
            (i.e. same format for all inherited classed). , by default False

        Returns
        -------
        Union[str, Any]
            Model response text (if return_details==False) or detailed response object.
        """
        pass


class OpenAIModel(LanguageModel):

    def __init__(self, api_key, model_version="gpt-4o-mini"):
        super().__init__(model_version)
        self.client = openai.OpenAI(
            api_key=api_key,
        )

    @retry_on_model_failure(max_retries=3)
    def make_forecast_with_probs(
        self, forecasting_question, context, verbose_reasoning=False, **kwargs
    ):
        response = self.query_model(
            forecasting_question, context, return_details=True, **kwargs
        )
        if verbose_reasoning:
            reply = response.choices[0].message.content
            print("Given answer was:\n", reply)
        top_logprobs = response.choices[0].logprobs.content[-1].top_logprobs
        top_tokens = []
        top_probs = []
        for top_logprob in top_logprobs:
            top_tokens.append(top_logprob.token)
            top_probs.append(np.exp(top_logprob.logprob))
        return top_tokens, top_probs

    def query_model(
        self, user_prompt: str, system_prompt: str, return_details=False, **kwargs
    ) -> Union[str, Any]:
        assert not any(
            key in kwargs for key in ["model", "messages", "logprobs"]
        ), "Invalid keyword argument"
        kwargs.setdefault("top_logprobs", 20)
        kwargs.setdefault("max_tokens", 512)
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


class AnthropicModel(LanguageModel):

    def __init__(self, api_key, model_version="claude-3-haiku-20240307"):
        super().__init__(model_version)
        self.client = anthropic.Anthropic(api_key=api_key)

    def query_model(
        self, user_prompt, system_prompt, return_details=False, **kwargs
    ) -> Union[str, Any]:
        assert not any(
            key in kwargs for key in ["model", "system", "messages"]
        ), "Invalid keyword argument"
        kwargs.setdefault("max_tokens", 512)
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

    @retry_on_model_failure(max_retries=3)
    def make_forecast_with_probs(
        self, forecasting_question, context, verbose_reasoning=False, **kwargs
    ):
        # TODO The API does not support logprobs, is the best solution to run the
        #   model multiple times?
        raise NotImplementedError("The API does not support logprobs")


class GeminiModel(LanguageModel):

    def __init__(self, api_key, model_version="gemini-1.5-flash-8b"):
        super().__init__(model_version)
        genai.configure(api_key=api_key)

    def query_model(
        self, user_prompt, system_prompt, return_details=False, **kwargs
    ) -> Union[str, Any]:
        assert not any(
            key in kwargs for key in ["response_logprobs"]
        ), "Invalid keyword argument"
        kwargs.setdefault("max_output_tokens", 512)
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

    @retry_on_model_failure(max_retries=3)
    def make_forecast_with_probs(
        self, forecasting_question, context, verbose_reasoning=False, **kwargs
    ):
        # the problem here is that even though we can access the logprobs, we
        # can get only the top 5 tokens and their probabilities, additionaly the
        # two digits of the number are not counted as one token TODO solve this
        raise NotImplementedError("The API does not support logprobs")


class XAIModel(LanguageModel):

    def __init__(self, api_key, model_version="grok-beta"):
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
        kwargs.setdefault("max_tokens", 512)
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

    @retry_on_model_failure(max_retries=3)
    def make_forecast_with_probs(
        self, forecasting_question, context, verbose_reasoning=False, **kwargs
    ):
        # TODO this does not work for some reason, should I try the CURL API
        # without using OpenAI or ist the API just broken? --> turns out, the
        # API does not return any logprobs, even if we query the model directly
        # using curl
        raise NotImplementedError("The API does not support logprobs")


class LLAMAModel(LanguageModel):

    def __init__(self, api_key, model_version="llama3.1-8b"):
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
        kwargs.setdefault("max_tokens", 512)
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

    @retry_on_model_failure(max_retries=3)
    def make_forecast_with_probs(
        self, forecasting_question, context, verbose_reasoning=False, **kwargs
    ):
        # for newer versions of the model (>= llama3.1-70b), the logprobs are
        # returned, but only for the tokens that are actually generated, not for the
        # top logprobs
        raise NotImplementedError("The API does not support logprobs")


class MistralModel(LanguageModel):

    def __init__(self, api_key, model_version="mistral-small-2409"):
        super().__init__(model_version)
        self.client = Mistral(api_key=api_key)

    def query_model(
        self, user_prompt, system_prompt, return_details=False, **kwargs
    ) -> Union[str, Any]:
        assert not any(
            key in kwargs for key in ["model", "messages", "logprobs"]
        ), "Invalid keyword argument"
        kwargs.setdefault("max_tokens", 512)
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

    @retry_on_model_failure(max_retries=3)
    def make_forecast_with_probs(
        self, forecasting_question, context, verbose_reasoning=False, **kwargs
    ):
        raise NotImplementedError("The API does not support logprobs")


class QwenModel(LanguageModel):

    def __init__(self, api_key, model_version="qwen-turbo"):
        super().__init__(model_version)
        dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"
        self.api_key = api_key

    def query_model(
        self, user_prompt, system_prompt, return_details=False, **kwargs
    ) -> Union[str, Any]:
        assert not any(
            key in kwargs for key in ["model", "messages"]
        ), "Invalid keyword argument"
        kwargs.setdefault("max_tokens", 512)
        kwargs.setdefault("seed", random.randint(1, 10000))
        response = dashscope.Generation.call(
            model=self.model_version,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            api_key=self.api_key**kwargs,
        )
        if return_details:
            return response
        else:
            return response.choices[0].message.content

    @retry_on_model_failure(max_retries=3)
    def make_forecast_with_probs(
        self, forecasting_question, context, verbose_reasoning=False, **kwargs
    ):
        raise NotImplementedError("The API does not support logprobs")
