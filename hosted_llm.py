import time
from typing import Optional

import requests


class HostedLLM:
    """
    OpenRouter wrapper for the Streamlit fraud-prevention prototype.

    This version:
    - accepts system_prompt, max_new_tokens and temperature per request;
    - works safely with the dynamic "openrouter/free" router;
    - does not send model-specific reasoning controls;
    - retries once with a larger output budget if a routed model uses the
      first budget without producing visible text;
    - returns useful OpenRouter error details.
    """

    def __init__(
        self,
        api_token: str,
        model: str = "openrouter/free",
        max_new_tokens: int = 1200,
        temperature: float = 0.0,
        timeout: int = 120,
    ):
        if not api_token or not str(api_token).strip():
            raise ValueError("OPENROUTER_API_KEY is missing or empty.")

        self.api_token = str(api_token).strip()
        self.model = model
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.timeout = int(timeout)

        self.endpoint_url = (
            "https://openrouter.ai/api/v1/chat/completions"
        )

    def _post(self, payload: dict) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
            "HTTP-Referer": (
                "https://payment-safety-graphrag.streamlit.app"
            ),
            "X-Title": "Payment Safety GraphRAG Research Prototype",
        }

        try:
            response = requests.post(
                self.endpoint_url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"OpenRouter network request failed: {e}"
            ) from e

        if not response.ok:
            raise RuntimeError(
                "OpenRouter request failed "
                f"({response.status_code}). "
                f"Response body: {response.text}"
            )

        try:
            return response.json()
        except ValueError as e:
            raise RuntimeError(
                "OpenRouter returned invalid JSON. "
                f"Response body: {response.text}"
            ) from e

    @staticmethod
    def _extract_text(result: dict):
        choices = result.get("choices") or []

        if not choices:
            return None, None

        choice = choices[0] or {}
        message = choice.get("message") or {}
        content = message.get("content")
        finish_reason = choice.get("finish_reason")

        if isinstance(content, str):
            content = content.strip()

        return content, finish_reason

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        if not prompt or not str(prompt).strip():
            raise ValueError("Prompt must not be empty.")

        messages = []

        if system_prompt and str(system_prompt).strip():
            messages.append(
                {
                    "role": "system",
                    "content": str(system_prompt).strip(),
                }
            )

        messages.append(
            {
                "role": "user",
                "content": str(prompt).strip(),
            }
        )

        requested_tokens = (
            int(max_new_tokens)
            if max_new_tokens is not None
            else self.max_new_tokens
        )

        # customer_guidance.py may request only 220/280 tokens.
        # That can be too small when openrouter/free selects a reasoning model.
        first_budget = max(
            requested_tokens,
            self.max_new_tokens,
            1200,
        )

        final_temperature = (
            float(temperature)
            if temperature is not None
            else self.temperature
        )

        # Do not send "reasoning" controls to openrouter/free.
        # It is a dynamic router and may select models with incompatible
        # reasoning settings.
        payload = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": first_budget,
            "temperature": final_temperature,
        }

        result = self._post(payload)
        content, finish_reason = self._extract_text(result)

        if content:
            return content

        # Retry once if the routed model exhausted the first output budget.
        if finish_reason == "length":
            retry_payload = dict(payload)
            retry_payload["max_completion_tokens"] = max(
                first_budget * 2,
                2000,
            )

            time.sleep(0.5)

            retry_result = self._post(retry_payload)
            retry_content, retry_finish_reason = self._extract_text(
                retry_result
            )

            if retry_content:
                return retry_content

            raise RuntimeError(
                "OpenRouter produced no visible answer after a retry. "
                f"First finish_reason={finish_reason}; "
                f"retry finish_reason={retry_finish_reason}. "
                f"Retry response: {retry_result}"
            )

        raise RuntimeError(
            "OpenRouter returned no visible text. "
            f"finish_reason={finish_reason}. "
            f"Full response: {result}"
        )
