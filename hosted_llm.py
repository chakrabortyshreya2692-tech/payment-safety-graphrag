import requests


class HostedLLM:

    def __init__(
        self,
        api_token,
        model="openrouter/free",
        max_new_tokens=300,
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

        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

        # ==============================================
        # Build messages
        # ==============================================

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

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": final_max_tokens,
            "temperature": final_temperature,
        }

        # ==============================================
        # Send request
        # ==============================================

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

        # ==============================================
        # Parse JSON
        # ==============================================

        try:

            result = response.json()

        except ValueError as e:

            raise RuntimeError(
                "OpenRouter returned invalid JSON: "
                f"{response.text}"
            ) from e

        # ==============================================
        # Validate response
        # ==============================================

        if (
            "choices" not in result
            or not result["choices"]
        ):

            raise RuntimeError(
                "OpenRouter returned no choices. "
                f"Full response: {result}"
            )

        message = (
            result["choices"][0]
            .get("message", {})
        )

        content = message.get(
            "content"
        )

        # ==============================================
        # Handle None/empty responses
        # ==============================================

        if content is None:

            raise RuntimeError(
                "OpenRouter returned content=None. "
                f"Full response: {result}"
            )

        if not isinstance(
            content,
            str,
        ):

            raise RuntimeError(
                "OpenRouter returned non-text content. "
                f"Full response: {result}"
            )

        content = content.strip()

        if not content:

            raise RuntimeError(
                "OpenRouter returned an empty response. "
                f"Full response: {result}"
            )

        return content
