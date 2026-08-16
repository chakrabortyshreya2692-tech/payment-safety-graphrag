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
    ):

        headers = {
            "Authorization": (
                f"Bearer {self.api_token}"
            ),
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,

            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],

            "max_tokens": self.max_new_tokens,

            "temperature": self.temperature,
        }

        response = requests.post(
            self.endpoint_url,
            headers=headers,
            json=payload,
            timeout=120,
        )

        response.raise_for_status()

        result = response.json()

        try:

            return (
                result["choices"][0]
                ["message"]["content"]
                .strip()
            )

        except (
            KeyError,
            IndexError,
            TypeError,
        ):

            raise RuntimeError(
                "Unexpected response from hosted LLM: "
                f"{result}"
            )
