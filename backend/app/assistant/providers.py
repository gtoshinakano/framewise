"""Chat providers and an OpenAI-compatible streaming client.

Providers resolve a model choice into an ``Endpoint``. ``local`` drives the
worker-owned Qwen 3.6 llama-server, ``openrouter`` uses the worker's API key,
and extra OpenAI-compatible servers (LM Studio, Ollama, vLLM, another
llama-server) can be listed in ``ASSISTANT_ENDPOINTS`` as JSON::

    [{"id": "lmstudio", "label": "LM Studio", "base_url": "http://192.168.0.10:1234/v1",
      "models": ["qwen3-14b"], "context": 32768, "api_key_env": "LMSTUDIO_KEY", "template_thinking": true}]
"""
import json
import os
import queue
import re
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field

from . import compact

OPENROUTER_URL = 'https://openrouter.ai/api/v1'
# Answers a turn that chose the local model while visual work holds the GPU (see ``runner.connect``).
FALLBACK_MODEL = 'openrouter/free'


class ProviderError(RuntimeError):
    """The provider rejected or failed a request; the message is user-facing."""


class Cancelled(Exception):
    """The user stopped the turn."""


@dataclass
class Endpoint:
    url: str
    model: str
    headers: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)
    context_limit: int | None = None
    timeout: float = 180
    fixed_model: bool = False  # Report ``model`` instead of the server's name (llama-server returns a file path).


@dataclass
class StepResult:
    content: str = ''
    reasoning: str = ''
    tool_calls: list = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict = field(default_factory=dict)
    model: str | None = None
    first_token_seconds: float | None = None
    seconds: float = 0.0


class Provider:
    id = ''
    label = ''
    thinking = True

    def describe(self):
        return {'id': self.id, 'label': self.label, 'available': self.available()[0], 'detail': self.available()[1],
                'models': self.models(), 'thinking': self.thinking, 'managed': False, 'free_model': False}

    def available(self):
        return True, ''

    def models(self):
        return []

    @contextmanager
    def session(self, model, thinking, cancel_check, on_stage):
        yield self.endpoint(model, thinking)

    def endpoint(self, model, thinking):
        raise NotImplementedError


class LocalQwen(Provider):
    id, label = 'local', 'Local · Qwen 3.6 35B'

    def describe(self):
        from .. import local_text_runtime
        info = super().describe()
        info.update(managed=True, context_limit=self.context_limit(), busy=bool(local_text_runtime.runtime and local_text_runtime.runtime.gpu_busy))
        info['models'] = [{'id': local_text_runtime.MODEL_ID, 'label': 'Qwen 3.6 35B A3B · Q4_K_M', 'context': self.context_limit()}]
        return info

    @staticmethod
    def context_limit():
        try:
            return int(os.environ.get('QWEN36_CONTEXT_SIZE', '65536'))
        except ValueError:
            return 65536

    def available(self):
        from .. import local_text_runtime
        if local_text_runtime.runtime is None:
            return False, 'Local model runtime is not running in this worker'
        return True, ''

    @contextmanager
    def session(self, model, thinking, cancel_check, on_stage, wait_for_gpu=True):
        """With ``wait_for_gpu=False`` a GPU held by visual work raises ``local_text_runtime.GpuBusy``."""
        from .. import db, local_text_runtime
        runtime = local_text_runtime.runtime
        if runtime is None:
            raise ProviderError('The local model runtime is not available in this worker')

        def keep_loaded():
            # Chat keeps the model resident between turns; the user can unload it.
            with db.connect() as conn:
                conn.execute('UPDATE local_text_model_state SET desired_loaded=1 WHERE id=1')

        if wait_for_gpu:
            keep_loaded()
        with runtime.chat_session(cancel_check, on_stage, wait=wait_for_gpu):
            if not wait_for_gpu:
                keep_loaded()  # Only once the GPU is ours; a fallback must not queue a model load behind visual work.
            loaded = runtime.process is not None and runtime.process.poll() is None
            if not loaded:
                on_stage('Loading local model')
            started = time.monotonic()
            try:
                url = runtime.ensure(cancel_check=cancel_check)
            except Cancelled:
                raise
            except Exception as exc:
                raise ProviderError(f'Local model could not start: {exc}') from exc
            if not url:
                raise ProviderError('Local model is not ready')
            runtime.publish('generating', 'Cut Assistance chat')
            endpoint = Endpoint(url + '/v1/chat/completions', local_text_runtime.MODEL_ID, context_limit=self.context_limit(), timeout=900, fixed_model=True,
                                extra={'chat_template_kwargs': {'enable_thinking': bool(thinking)}, 'cache_prompt': True, 'top_p': 0.95, 'top_k': 20})
            endpoint.extra['_load_seconds'] = 0 if loaded else round(time.monotonic() - started, 2)
            try:
                yield endpoint
            finally:
                if runtime.process and runtime.process.poll() is None:
                    runtime.publish('loaded', 'Ready')


_openrouter_context = {'at': 0.0, 'models': {}}
_openrouter_lock = threading.Lock()


def openrouter_context(model):
    """Context length from OpenRouter's public model list, cached for an hour."""
    with _openrouter_lock:
        if time.time() - _openrouter_context['at'] > 3600:
            try:
                with urllib.request.urlopen(OPENROUTER_URL + '/models', timeout=10) as response:
                    data = json.load(response).get('data') or []
                _openrouter_context['models'] = {item['id']: item.get('context_length') for item in data if isinstance(item, dict) and item.get('id')}
            except (OSError, ValueError, urllib.error.URLError):
                pass
            _openrouter_context['at'] = time.time()
        value = _openrouter_context['models'].get(model)
        return value if isinstance(value, int) else None


class OpenRouter(Provider):
    id, label = 'openrouter', 'OpenRouter'

    def describe(self):
        info = super().describe()
        info['free_model'] = True
        info['fallback_model'] = FALLBACK_MODEL  # The UI offers the GPU-busy fallback only when the worker publishes this.
        return info

    def available(self):
        if not os.environ.get('OPENROUTER_API_KEY', '').strip():
            return False, 'OPENROUTER_API_KEY is not configured for the inference worker'
        return True, ''

    def endpoint(self, model, thinking):
        ok, detail = self.available()
        if not ok:
            raise ProviderError(detail)
        extra = {'usage': {'include': True}}
        if thinking:
            extra['reasoning'] = {'enabled': True}
        return Endpoint(OPENROUTER_URL + '/chat/completions', model, extra=extra, context_limit=openrouter_context(model),
                        headers={'Authorization': f"Bearer {os.environ['OPENROUTER_API_KEY'].strip()}", 'X-Title': 'Framewise Cut Assistance'})


class OpenAICompatible(Provider):
    def __init__(self, config):
        self.id = config['id']
        self.label = config.get('label') or config['id']
        self.base_url = config['base_url'].rstrip('/')
        self.model_ids = [str(model) for model in config.get('models') or []]
        self.context = config.get('context') if isinstance(config.get('context'), int) else None
        self.key_env = config.get('api_key_env')
        self.template_thinking = bool(config.get('template_thinking'))

    def describe(self):
        info = super().describe()
        info['models'] = [{'id': model, 'label': model, 'context': self.context} for model in self.model_ids]
        info['free_model'] = not self.model_ids
        info['context_limit'] = self.context
        return info

    def endpoint(self, model, thinking):
        headers = {}
        if self.key_env and os.environ.get(self.key_env):
            headers['Authorization'] = f'Bearer {os.environ[self.key_env]}'
        extra = {'chat_template_kwargs': {'enable_thinking': bool(thinking)}} if self.template_thinking else {}
        return Endpoint(self.base_url + '/chat/completions', model, headers=headers, extra=extra, context_limit=self.context, timeout=600)


def registry():
    providers = {'local': LocalQwen(), 'openrouter': OpenRouter()}
    try:
        configured = json.loads(os.environ.get('ASSISTANT_ENDPOINTS') or '[]')
    except ValueError:
        configured = []
    for item in configured if isinstance(configured, list) else []:
        if isinstance(item, dict) and re.fullmatch(r'[a-z][a-z0-9_-]{1,30}', str(item.get('id', ''))) and item['id'] not in providers and str(item.get('base_url', '')).startswith(('http://', 'https://')):
            providers[item['id']] = OpenAICompatible(item)
    return providers


# ----------------------------------------------------------------- streaming

class ThinkSplitter:
    """Route inline <think>…</think> blocks to reasoning for servers that do not parse them."""

    def __init__(self):
        self.mode = 'start'
        self.buffer = ''

    def feed(self, text):
        self.buffer += text
        content, reasoning = '', ''
        while self.buffer:
            if self.mode == 'start':
                stripped = self.buffer.lstrip()
                if not stripped:
                    return content, reasoning
                if stripped.startswith('<think>'):
                    self.buffer = stripped[len('<think>'):]
                    self.mode = 'think'
                    continue
                if '<think>'.startswith(stripped):
                    return content, reasoning  # Wait for more characters.
                self.mode = 'content'
            if self.mode == 'think':
                end = self.buffer.find('</think>')
                if end >= 0:
                    reasoning += self.buffer[:end]
                    self.buffer = self.buffer[end + len('</think>'):].lstrip('\n')
                    self.mode = 'content'
                    continue
                keep = next((n for n in range(min(len(self.buffer), 8), 0, -1) if '</think>'.startswith(self.buffer[-n:])), 0)
                reasoning += self.buffer[:len(self.buffer) - keep]
                self.buffer = self.buffer[len(self.buffer) - keep:]
                return content, reasoning
            content += self.buffer
            self.buffer = ''
        return content, reasoning

    def flush(self):
        rest, self.buffer = self.buffer, ''
        return ('', rest) if self.mode == 'think' else (rest, '')


def normalize_usage(usage, timings=None):
    out = {}
    if isinstance(usage, dict):
        for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            if isinstance(usage.get(key), int):
                out[key] = usage[key]
        details = usage.get('completion_tokens_details') or {}
        if isinstance(details.get('reasoning_tokens'), int):
            out['reasoning_tokens'] = details['reasoning_tokens']
        prompt_details = usage.get('prompt_tokens_details') or {}
        if isinstance(prompt_details.get('cached_tokens'), int):
            out['cached_tokens'] = prompt_details['cached_tokens']
        if isinstance(usage.get('cost'), (int, float)):
            out['cost'] = usage['cost']
    if isinstance(timings, dict):
        prompt_n, cache_n, predicted_n = timings.get('prompt_n'), timings.get('cache_n'), timings.get('predicted_n')
        if 'prompt_tokens' not in out and isinstance(prompt_n, int):
            out['prompt_tokens'] = prompt_n + (cache_n if isinstance(cache_n, int) else 0)
        if 'completion_tokens' not in out and isinstance(predicted_n, int):
            out['completion_tokens'] = predicted_n
        if isinstance(cache_n, int) and 'cached_tokens' not in out:
            out['cached_tokens'] = cache_n
        if isinstance(timings.get('predicted_per_second'), (int, float)):
            out['tokens_per_second'] = round(timings['predicted_per_second'], 1)
        if isinstance(timings.get('prompt_per_second'), (int, float)):
            out['prompt_tokens_per_second'] = round(timings['prompt_per_second'], 1)
    if 'total_tokens' not in out and 'prompt_tokens' in out and 'completion_tokens' in out:
        out['total_tokens'] = out['prompt_tokens'] + out['completion_tokens']
    return out


def error_message(raw, status=None):
    try:
        data = json.loads(raw)
        error = data.get('error') if isinstance(data, dict) else None
        message = error.get('message') if isinstance(error, dict) else error if isinstance(error, str) else None
        metadata = error.get('metadata') if isinstance(error, dict) else None
        if isinstance(metadata, dict) and isinstance(metadata.get('raw'), str):
            message = f"{message}: {metadata['raw'][:300]}"
    except (TypeError, ValueError):
        message = raw.strip()[:400] if isinstance(raw, str) else None
    prefix = f'HTTP {status}: ' if status else ''
    return prefix + (message or 'request failed')


def stream_chat(endpoint, messages, tools, *, max_tokens, temperature, on_delta, checkpoint, on_tool=None, poll=0.25):
    """Stream one completion. ``checkpoint`` raises to cancel; ``on_delta`` gets text pieces."""
    payload = {'model': endpoint.model, 'messages': messages, 'stream': True, 'stream_options': {'include_usage': True},
               'max_tokens': max_tokens, 'temperature': temperature}
    if tools:
        payload['tools'] = tools
        payload['tool_choice'] = 'auto'
    payload.update({key: value for key, value in endpoint.extra.items() if not key.startswith('_')})
    request = urllib.request.Request(endpoint.url, data=json.dumps(payload, ensure_ascii=False).encode(), method='POST',
                                     headers={'Content-Type': 'application/json', 'Accept': 'text/event-stream', **endpoint.headers})
    lines = queue.Queue()
    holder = {}

    def read():
        try:
            response = urllib.request.urlopen(request, timeout=endpoint.timeout)
            holder['response'] = response
            for raw in response:
                lines.put(('line', raw.decode('utf-8', 'replace')))
            lines.put(('end', None))
        except urllib.error.HTTPError as exc:
            try:
                body = exc.read().decode('utf-8', 'replace')
            except OSError:
                body = ''
            finally:
                exc.close()
            lines.put(('error', ProviderError(error_message(body, exc.code))))
        except Exception as exc:  # Network failures surface to the turn.
            if not holder.get('closed'):
                lines.put(('error', ProviderError(f'Connection failed: {exc or type(exc).__name__}')))

    started = time.monotonic()
    result = StepResult()
    calls = {}
    splitter = ThinkSplitter()
    timings = None
    thread = threading.Thread(target=read, daemon=True, name='assistant-stream')
    thread.start()
    try:
        while True:
            try:
                kind, value = lines.get(timeout=poll)
            except queue.Empty:
                checkpoint()
                continue
            if kind == 'error':
                raise value
            if kind == 'end':
                break
            line = value.strip()
            if not line.startswith('data:'):
                continue
            data = line[5:].strip()
            if data == '[DONE]':
                break
            try:
                chunk = json.loads(data)
            except ValueError:
                continue
            if isinstance(chunk.get('error'), (dict, str)):
                raise ProviderError(error_message(data))
            if chunk.get('model'):
                result.model = chunk['model']
            if chunk.get('usage'):
                result.usage = chunk['usage']
            if chunk.get('timings'):
                timings = chunk['timings']
            for choice in chunk.get('choices') or []:
                delta = choice.get('delta') or {}
                reasoning = delta.get('reasoning_content') or delta.get('reasoning') or ''
                content_piece = delta.get('content') or ''
                if content_piece:
                    content_piece, extra_reasoning = splitter.feed(content_piece)
                    reasoning += extra_reasoning
                for call in delta.get('tool_calls') or []:
                    index = call.get('index', len(calls))
                    entry = calls.setdefault(index, {'id': '', 'name': '', 'arguments': ''})
                    entry['id'] = call.get('id') or entry['id']
                    function = call.get('function') or {}
                    named = bool(entry['name'])
                    entry['name'] += function.get('name') or ''
                    entry['arguments'] += function.get('arguments') or ''
                    if on_tool and entry['name'] and not named:
                        on_tool(entry['name'])
                if (reasoning or content_piece or delta.get('tool_calls')) and result.first_token_seconds is None:
                    result.first_token_seconds = round(time.monotonic() - started, 2)
                if reasoning or content_piece:
                    result.reasoning += reasoning
                    result.content += content_piece
                    on_delta(content_piece, reasoning)
                if choice.get('finish_reason'):
                    result.finish_reason = choice['finish_reason']
            checkpoint()
    finally:
        holder['closed'] = True
        response = holder.get('response')
        if response is not None:
            try:
                response.close()
            except Exception:
                pass
    tail_content, tail_reasoning = splitter.flush()
    if tail_content or tail_reasoning:
        result.content += tail_content
        result.reasoning += tail_reasoning
        on_delta(tail_content, tail_reasoning)
    for index in sorted(calls):
        entry = calls[index]
        if entry['name']:
            result.tool_calls.append({'id': entry['id'] or f'call_{index}_{int(time.time() * 1000) % 100000}', 'name': entry['name'].strip(), 'arguments': entry['arguments'] or '{}'})
    result.usage = normalize_usage(result.usage, timings)
    if result.reasoning and 'reasoning_tokens' not in result.usage:
        result.usage['reasoning_tokens_estimated'] = compact.estimate_tokens(result.reasoning)
    result.seconds = round(time.monotonic() - started, 2)
    return result
