import requests


class HostedLLM:

    def __init__(
        self,
        endpoint_url,
        api_token,
        max_new_tokens=300,
        temperature=0.0,
    ):

        self.endpoint_url = endpoint_url.rstrip("/")
        self.api_token = api_token
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature


    def generate(
        self,
        prompt,
    ):

        url = (
            self.endpoint_url
            + "/v1/chat/completions"
        )

        headers = {
            "Authorization":
                f"Bearer {self.api_token}",

            "Content-Type":
                "application/json",
        }


        payload = {

            "model": "tgi",

            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],

            "max_tokens":
                self.max_new_tokens,

            "temperature":
                self.temperature,

            "stream":
                False,
        }


        response = requests.post(
            url,
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
                "Unexpected response received "
                "from hosted LLM endpoint: "
                f"{result}"
            )
