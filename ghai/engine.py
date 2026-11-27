"""GHAi çıkarım motoru (llama.cpp üzerinden, CPU'da da çalışır)."""
import os

from llama_cpp import Llama

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_MODEL = os.environ.get("GHAI_MODEL", os.path.join(ROOT, "dist", "GHAi-3B-Q4_K_M.gguf"))


class GHAiEngine:
    def __init__(self, model_path=DEFAULT_MODEL, n_ctx=8192, n_threads=None, n_gpu_layers=-1):
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads or os.cpu_count(),
                         n_gpu_layers=n_gpu_layers, verbose=False)

    def _messages(self, history, system):
        msgs = [{"role": "system", "content": system}] if system else []
        return msgs + list(history)

    def stream(self, history, system=None, temperature=0.3, max_tokens=2048):
        for chunk in self.llm.create_chat_completion(
            self._messages(history, system), temperature=temperature, top_p=0.95,
            repeat_penalty=1.05, max_tokens=max_tokens, stream=True,
        ):
            delta = chunk["choices"][0]["delta"].get("content")
            if delta:
                yield delta

    def reply(self, history, system=None, **kw):
        return "".join(self.stream(history, system=system, **kw))
