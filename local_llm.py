from __future__ import annotations
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalLLM:
    def __init__(
        self,
        model_path,
        max_new_tokens=350,
        temperature=0.0,
        top_p=0.9,
        prefer_4bit=True,
    ):
        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"Model directory not found: {model_path}. "
                "Update MODEL_DIR in Section 1."
            )

        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )

        use_cuda = torch.cuda.is_available()
        kwargs = {
            "local_files_only": True,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        quantized = False
        if use_cuda and prefer_4bit:
            try:
                from transformers import BitsAndBytesConfig
                import bitsandbytes  # noqa
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                kwargs["device_map"] = "auto"
                quantized = True
            except Exception:
                quantized = False

        if not quantized:
            if use_cuda:
                kwargs["device_map"] = "auto"
                kwargs["torch_dtype"] = (
                    torch.bfloat16
                    if torch.cuda.is_bf16_supported()
                    else torch.float16
                )
            else:
                kwargs["torch_dtype"] = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **kwargs,
        )
        self.model.eval()

        mode = "4-bit CUDA" if quantized else "CUDA" if use_cuda else "CPU"
        print(f"Local Qwen model loaded successfully ({mode}).")

    def generate(
        self,
        prompt,
        system_prompt=None,
        max_new_tokens=None,
        temperature=None,
    ):
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": prompt.strip()})

        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            add_special_tokens=False,
        )

        device = next(self.model.parameters()).device
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

        temp = self.temperature if temperature is None else float(temperature)
        max_tokens = (
            self.max_new_tokens
            if max_new_tokens is None
            else int(max_new_tokens)
        )
        do_sample = temp > 0

        generation_args = {
            **model_inputs,
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if do_sample:
            generation_args["temperature"] = temp
            generation_args["top_p"] = self.top_p

        with torch.inference_mode():
            output = self.model.generate(**generation_args)

        input_len = model_inputs["input_ids"].shape[1]
        return self.tokenizer.decode(
            output[0, input_len:],
            skip_special_tokens=True,
        ).strip()
