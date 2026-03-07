import warnings

# Third-party deprecation warnings from upstream libs; harmless for runtime behavior.
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="`torch.nn.utils.weight_norm` is deprecated.*",
    category=FutureWarning,
)

from .config import Config

__all__ = ["Config", "STT", "TTS", "LLM", "get_tools"]


def __getattr__(name):
    if name == "STT":
        from .stt import STT

        return STT
    if name == "TTS":
        from .tts import TTS

        return TTS
    if name == "LLM":
        from .llm import LLM

        return LLM
    if name == "get_tools":
        from .utility import get_tools

        return get_tools
    raise AttributeError(f"module 'narubot' has no attribute '{name}'")
