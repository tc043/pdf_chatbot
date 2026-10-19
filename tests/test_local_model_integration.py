"""Exercise the real local-generation path against real `transformers` code.

Uses a tiny randomly initialised GPT-2 and a hand-built word-level tokenizer, so nothing
is downloaded and the output is gibberish; what is verified is the plumbing that the
fake-based unit tests cannot cover: chat-template handling, TextIteratorStreamer,
StoppingCriteria-based cancellation, and error propagation out of ``model.generate``.

Skipped automatically when torch / transformers are not installed (they are optional).
"""

from __future__ import annotations

import threading
import time

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402

from app import config, llm_engine  # noqa: E402
from app.exceptions import GenerationError  # noqa: E402

pytestmark = pytest.mark.local_model

WORDS = [f"w{i}" for i in range(60)]
CHAT_TEMPLATE = (
    "{% for m in messages %}{{ m['content'] }} {% endfor %}"
    "{% if enable_thinking is defined and not enable_thinking %}nothink {% endif %}"
    "{% if add_generation_prompt %}answer: {% endif %}"
)


@pytest.fixture(scope="module")
def tiny_model():
    specials = ["[UNK]", "[PAD]", "[EOS]"]
    vocab = {token: i for i, token in enumerate(specials + WORDS + ["answer:", "nothink", "hello"])}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]"
    )
    tokenizer.chat_template = CHAT_TEMPLATE

    torch.manual_seed(0)
    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=len(vocab), n_embd=16, n_layer=1, n_head=2, n_positions=4096
        )
    ).eval()
    return model, tokenizer


@pytest.fixture
def manager(tiny_model):
    model, tokenizer = tiny_model
    mgr = llm_engine.LocalModelManager.get_instance()  # conftest resets the singleton per test
    mgr.model, mgr.tokenizer = model, tokenizer
    assert mgr.is_loaded  # so no weights are ever downloaded
    return mgr


SETTINGS = config.Settings(local_token_timeout_s=30)


def stream(prompt="hello w1 w2", **kwargs):
    kwargs.setdefault("settings", SETTINGS)
    return llm_engine.generate_local_stream(prompt, **kwargs)


def test_streams_text_and_finishes(manager):
    pieces = list(stream(max_tokens=20, temperature=0.0))
    assert pieces and all(isinstance(p, str) and p for p in pieces)
    assert not manager.generation_lock.locked()  # released after a normal finish


def test_thinking_flag_reaches_the_chat_template(manager, monkeypatch):
    seen = []
    real = manager.tokenizer.apply_chat_template

    def spy(messages, **kwargs):
        seen.append(kwargs.get("enable_thinking"))
        return real(messages, **kwargs)

    monkeypatch.setattr(manager.tokenizer, "apply_chat_template", spy)
    list(stream(max_tokens=3, enable_thinking=False))
    list(stream(max_tokens=3, enable_thinking=True))
    assert seen == [False, True]


def test_sampling_and_greedy_modes_both_work(manager):
    assert list(stream(max_tokens=8, temperature=0.0))  # greedy: no temperature kwarg passed
    assert list(stream(max_tokens=8, temperature=0.9))


def test_abandoned_stream_stops_generation_and_frees_the_model(manager, monkeypatch):
    """A client that disconnects mid-answer must not leave the CPU busy / the model locked."""
    budget = 3000
    real_generate = manager.model.generate
    produced = []

    def spy(**kwargs):
        output = real_generate(**kwargs)
        produced.append(output.shape[1] - kwargs["input_ids"].shape[1])
        return output

    monkeypatch.setattr(manager.model, "generate", spy)

    generator = stream(max_tokens=budget, temperature=0.0)
    next(generator)  # generation is now running with a 3000-token budget
    generator.close()  # the client disconnects

    assert produced, "generate() was still running after the stream was closed"
    assert produced[0] < budget // 10, f"generated {produced[0]} tokens instead of stopping early"
    assert not manager.generation_lock.locked()
    assert not [t for t in threading.enumerate() if t.name == "local-generate"]


def test_generate_crash_surfaces_as_error_not_a_hang(manager, monkeypatch):
    def broken_generate(**kwargs):
        raise RuntimeError("simulated out-of-memory")

    monkeypatch.setattr(manager.model, "generate", broken_generate)
    started = time.monotonic()
    with pytest.raises(GenerationError, match="failed while generating"):
        list(stream(max_tokens=5))
    assert time.monotonic() - started < 10
    assert not manager.generation_lock.locked()


def test_concurrent_requests_are_serialised_not_interleaved(manager):
    results, errors = [], []

    def worker():
        try:
            results.append("".join(stream(max_tokens=10, temperature=0.0)))
        except Exception as exc:  # pragma: no cover - would indicate a thread-safety bug
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors
    assert len(results) == 3 and len(set(results)) == 1  # greedy => identical outputs


def test_a_busy_model_gives_queued_requests_a_clear_error_instead_of_racing(manager):
    """If another generation holds the model (e.g. a stalled one), a new request must wait a
    bounded time and then fail clearly rather than call the model concurrently."""
    assert manager.generation_lock.acquire(timeout=1)
    try:
        started = time.monotonic()
        with pytest.raises(GenerationError, match="busy"):
            list(stream(settings=config.Settings(local_token_timeout_s=1)))
        assert 0.8 < time.monotonic() - started < 5
    finally:
        manager.generation_lock.release()
    assert list(stream(max_tokens=3))  # and it works again once the model is free
