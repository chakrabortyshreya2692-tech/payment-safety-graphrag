import requests


class HostedLLM:

    def __init__(
        self,
        api_token,
        model="openrouter/free",
        max_new_tokens=700,
        temperature=0.0,
    ):
        self.api_token = api_token
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self.endpoint_url = (
            "https://openrouter.ai/api/v1/chat/completions"
        )

    def generate(
        self,
        prompt,
        system_prompt=None,
        max_new_tokens=None,
        temperature=None,
    ):

        # -------------------------------------------------
        # Headers
        # -------------------------------------------------

        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

        # -------------------------------------------------
        # Build messages
        # -------------------------------------------------

        messages = []

        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        # -------------------------------------------------
        # Request-specific generation settings
        # -------------------------------------------------

        final_max_tokens = (
            max_new_tokens
            if max_new_tokens is not None
            else self.max_new_tokens
        )

        final_temperature = (
            temperature
            if temperature is not None
            else self.temperature
        )

        # -------------------------------------------------
        # OpenRouter request payload
        # -------------------------------------------------

        payload = {
            "model": self.model,
            "messages": messages,

            # Enough output budget for reasoning + answer
            "max_completion_tokens": final_max_tokens,

            "temperature": final_temperature,

            },
        }

        # -------------------------------------------------
        # Send request
        # -------------------------------------------------

        try:
            response = requests.post(
                self.endpoint_url,
                headers=headers,
                json=payload,
                timeout=120,
            )

            response.raise_for_status()

        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"OpenRouter request failed: {e}"
            ) from e

        # -------------------------------------------------
        # Parse response JSON
        # -------------------------------------------------

        try:
            result = response.json()

        except ValueError as e:
            raise RuntimeError(
                "OpenRouter returned invalid JSON. "
                f"Response body: {response.text}"
            ) from e

        # -------------------------------------------------
        # Check response structure
        # -------------------------------------------------

        if (
            "choices" not in result
            or not result["choices"]
        ):
            raise RuntimeError(
                "OpenRouter returned no choices. "
                f"Full response: {result}"
            )

        choice = result["choices"][0]

        message = choice.get(
            "message",
            {}
        )

        finish_reason = choice.get(
            "finish_reason"
        )

        content = message.get(
            "content"
        )

        # -------------------------------------------------
        # Handle token-limit issue
        # -------------------------------------------------

        if (
            finish_reason == "length"
            and not content
        ):
            raise RuntimeError(
                "The hosted LLM exhausted its completion "
                "token budget before producing the final "
                "answer. Increase max_new_tokens or reduce "
                "reasoning effort."
            )

        # -------------------------------------------------
        # Handle missing content
        # -------------------------------------------------

        if content is None:
            raise RuntimeError(
                "OpenRouter returned no final text. "
                f"Finish reason: {finish_reason}. "
                f"Full response: {result}"
            )

        # -------------------------------------------------
        # Make sure the content is text
        # -------------------------------------------------

        if not isinstance(
            content,
            str,
        ):
            raise RuntimeError(
                "OpenRouter returned non-text content. "
                f"Full response: {result}"
            )

        content = content.strip()

        # -------------------------------------------------
        # Handle empty response
        # -------------------------------------------------

        if not content:
            raise RuntimeError(
                "OpenRouter returned an empty final response. "
                f"Finish reason: {finish_reason}. "
                f"Full response: {result}"
            )

        # -------------------------------------------------
        # Successful response
        # -------------------------------------------------

        return content
