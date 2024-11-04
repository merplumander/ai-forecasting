from abc import ABC, abstractmethod
from typing import Tuple

import anthropic
import google.generativeai as genai
import numpy as np
import openai

from src.query.utils import extract_probability, retry_on_model_failure


class LanguageModel(ABC):

    def __init__(self, model_version):
        self.model_version = model_version

    @abstractmethod
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
        pass

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
    def _query_model(self, forecasting_question: str, context: str, **kwargs):
        pass


class OpenAIModel(LanguageModel):

    def __init__(self, api_key, model_version="gpt-3.5-turbo"):
        super().__init__(model_version)
        self.client = openai.OpenAI(
            api_key=api_key,
        )

    @retry_on_model_failure(max_retries=3)
    def make_forecast(
        self,
        forecasting_question,
        context,
        verbose_reasoning=False,
        **kwargs,
    ):
        response = self._query_model(forecasting_question, context, **kwargs)
        reply = response.choices[0].message.content
        if verbose_reasoning:
            print("Given answer was:\n", reply)
        return (extract_probability(reply), reply)

    @retry_on_model_failure(max_retries=3)
    def make_forecast_with_probs(
        self, forecasting_question, context, verbose_reasoning=False, **kwargs
    ):
        response = self._query_model(forecasting_question, context, **kwargs)
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

    def _query_model(self, forecasting_question: str, context: str, **kwargs):
        assert not any(
            key in kwargs for key in ["model", "messages", "logprobs"]
        ), "Invalid keyword argument"
        kwargs.setdefault("top_logprobs", 20)
        kwargs.setdefault("max_tokens", 512)
        response = self.client.chat.completions.create(
            model=self.model_version,
            messages=[
                {"role": "user", "content": forecasting_question},
                {"role": "system", "content": context},
            ],
            logprobs=True,
            **kwargs,
        )
        return response


class AnthropicModel(LanguageModel):

    def __init__(self, api_key, model_version="claude-3-haiku-20240307"):
        super().__init__(model_version)
        self.client = anthropic.Anthropic(api_key=api_key)

    @retry_on_model_failure(max_retries=3)
    def make_forecast(
        self,
        forecasting_question,
        context,
        verbose_reasoning=False,
        **kwargs,
    ):
        response = self._query_model(forecasting_question, context, **kwargs)
        reply = response.content[0].text
        if verbose_reasoning:
            print("Given answer was:\n", reply)
        return (extract_probability(reply), reply)

    def _query_model(self, forecasting_question, context, **kwargs):
        assert not any(
            key in kwargs for key in ["model", "system", "messages"]
        ), "Invalid keyword argument"
        kwargs.setdefault("max_tokens", 512)
        response = self.client.messages.create(
            model=self.model_version,
            system=context,
            messages=[
                {"role": "user", "content": f"{forecasting_question}"},
            ],
            **kwargs,
        )
        return response

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

    @retry_on_model_failure(max_retries=3)
    def make_forecast(
        self,
        forecasting_question,
        context,
        verbose_reasoning=False,
        **kwargs,
    ):
        response = self._query_model(forecasting_question, context, **kwargs)
        if verbose_reasoning:
            print("Given answer was:\n", response.text)
        return (extract_probability(response.text), response.text)

    def _query_model(self, forecasting_question, context, **kwargs):
        assert not any(
            key in kwargs for key in ["response_logprobs"]
        ), "Invalid keyword argument"
        kwargs.setdefault("max_output_tokens", 512)
        # kwargs.setdefault("logprobs", 5)
        config = genai.GenerationConfig(response_logprobs=False, **kwargs)
        model = genai.GenerativeModel(
            model_name=self.model_version,
            generation_config=config,
            system_instruction=context,
        )
        response = model.generate_content(forecasting_question)
        return response

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

    @retry_on_model_failure(max_retries=3)
    def make_forecast(
        self,
        forecasting_question,
        context,
        verbose_reasoning=False,
        **kwargs,
    ):
        response = self._query_model(forecasting_question, context, **kwargs)
        reply = response.choices[0].message.content
        if verbose_reasoning:
            print("Given answer was:\n", reply)
        return (extract_probability(reply), reply)

    def _query_model(self, forecasting_question, context, **kwargs):
        assert not any(
            key in kwargs for key in ["model", "messages", "logprobs"]
        ), "Invalid keyword argument"
        kwargs.setdefault("max_tokens", 512)
        response = self.client.chat.completions.create(
            model=self.model_version,
            messages=[
                {"role": "user", "content": f"{forecasting_question}"},
                {"role": "system", "content": f"{context}"},
            ],
            **kwargs,
        )
        return response

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

    @retry_on_model_failure(max_retries=3)
    def make_forecast(
        self,
        forecasting_question,
        context,
        verbose_reasoning=False,
        **kwargs,
    ):
        response = self._query_model(forecasting_question, context, **kwargs)
        reply = response.choices[0].message.content
        if verbose_reasoning:
            print("Given answer was:\n", reply)
        return (extract_probability(reply), reply)

    def _query_model(self, forecasting_question, context, **kwargs):
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
                    "content": context,
                },
                {"role": "user", "content": forecasting_question},
            ],
            logprobs=True,
            **kwargs,
        )
        return response

    @retry_on_model_failure(max_retries=3)
    def make_forecast_with_probs(
        self, forecasting_question, context, verbose_reasoning=False, **kwargs
    ):
        # for newer versions of the model (>= llama3.1-70b), the logprobs are
        # returned, but only for the tokens that are actually generated, not for the
        # top logprobs
        raise NotImplementedError("The API does not support logprobs")
