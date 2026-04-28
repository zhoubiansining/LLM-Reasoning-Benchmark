import os
import asyncio
from time import sleep

try:
    import openai
    from openai import OpenAI, AsyncOpenAI
except ImportError as e:
    pass

from lcb_runner.lm_styles import LMStyle
from lcb_runner.runner.base_runner import BaseRunner


class OpenAIRunner(BaseRunner):
    client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_KEY") or "dummy-key",
        base_url=os.getenv("OPENAI_BASE_URL") or "http://127.0.0.1:8000/v1",
    )

    def __init__(self, args, model):
        super().__init__(args, model)
        request_model_name = model.model_name

        if model.model_style == LMStyle.OpenAIReasonPreview:
            self.client_kwargs: dict[str | str] = {
                "model": request_model_name,
                "max_completion_tokens": 25000,
            }
        elif model.model_style == LMStyle.OpenAIReason:
            assert (
                "__" in args.model
            ), f"Model {args.model} is not a valid OpenAI Reasoning model as we require reasoning effort in model name."
            _, reasoning_effort = args.model.split("__", 1)
            self.client_kwargs: dict[str | str] = {
                "model": request_model_name,
                "reasoning_effort": reasoning_effort,
            }
        else:
            self.client_kwargs: dict[str | str] = {
                "model": request_model_name,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "top_p": args.top_p,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "n": args.n,
                "timeout": args.openai_timeout,
                # "stop": args.stop, --> stop is only used for base models currently
            }

    async def _run_single_async(self, prompt: list[dict[str, str]], n: int = 10) -> list[str]:
        assert isinstance(prompt, list)

        if n == 0:
            print("Max retries reached. Returning empty response.")
            return []

        try:
            response = await OpenAIRunner.client.chat.completions.create(
                messages=prompt,
                **self.client_kwargs,
            )
        except (
            openai.APIError,
            openai.RateLimitError,
            openai.InternalServerError,
            openai.OpenAIError,
            openai.APIStatusError,
            openai.APITimeoutError,
            openai.InternalServerError,
            openai.APIConnectionError,
        ) as e:
            print("Exception: ", repr(e))
            print("Sleeping for 30 seconds...")
            print("Consider reducing the number of parallel processes.")
            sleep(30)
            return await self._run_single_async(prompt, n=n - 1)
        except Exception as e:
            print(f"Failed to run the model for {prompt}!")
            print("Exception: ", repr(e))
            raise e

        return [
            c.message.content
            or (c.message.reasoning_content if hasattr(c.message, "reasoning_content") else "")
            for c in response.choices
        ]

    def _run_single(self, prompt: list[dict[str, str]], n: int = 10) -> list[str]:
        return asyncio.run(self._run_single_async(prompt, n=n))
