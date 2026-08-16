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
        """
        Generate a response from OpenRouter.

        Parameters may override the defaults configured when
        HostedLLM is instantiated.
        """

        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

        # -----------------------------------------------
        # Construct messages
        # -----------------------------------------------

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

        # -----------------------------------------------
        # Allow per-request settings
        # -----------------------------------------------

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

        # -----------------------------------------------
        # Call OpenRouter
        # -----------------------------------------------

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

        # -----------------------------------------------
        # Parse response
        # -----------------------------------------------

        try:
            result = response.json()

            content = (
                result["choices"][0]
                ["message"]["content"]
            )

            if not content:
                raise RuntimeError(
                    "OpenRouter returned an empty response."
                )

            return content.strip()

        except (
            KeyError,
            IndexError,
            TypeError,
            ValueError,
        ) as e:

            raise RuntimeError(
                "Unexpected response from hosted LLM: "
                f"{response.text}"
            ) from e
