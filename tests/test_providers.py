from rag_eval_sdk.providers import (
    DiskCachedGenerator,
    GenerationResult,
    OpenAICompatibleProvider,
)


class CountingGenerator:
    name = "counting"

    def __init__(self):
        self.calls = 0

    def generate(self, messages, *, temperature=0.0, max_tokens=512):
        self.calls += 1
        return GenerationResult("answer", self.name, 0.01)


def test_disk_cache_avoids_repeated_provider_call(tmp_path):
    underlying = CountingGenerator()
    cached = DiskCachedGenerator(underlying, tmp_path)
    messages = [{"role": "user", "content": "question"}]
    first = cached.generate(messages)
    second = cached.generate(messages)
    assert first.cached is False
    assert second.cached is True
    assert underlying.calls == 1


def test_provider_presets_leave_model_and_rate_explicit():
    groq = OpenAICompatibleProvider.groq(
        model="caller-selected-model", requests_per_minute=12
    )
    nim = OpenAICompatibleProvider.nvidia_nim(
        model="caller-selected-model", requests_per_minute=7
    )
    assert groq.base_url == "https://api.groq.com/openai/v1"
    assert groq.api_key_env == "GROQ_API_KEY"
    assert groq.model == "caller-selected-model"
    assert groq.requests_per_minute == 12
    assert nim.base_url == "https://integrate.api.nvidia.com/v1"
    assert nim.api_key_env == "NVIDIA_API_KEY"
    assert nim.requests_per_minute == 7
