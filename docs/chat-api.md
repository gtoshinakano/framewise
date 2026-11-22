# Chat API (OpenAI-compatible)

Other machines on the LAN can use this server as a chat provider backed by Qwen 3.6. The API process
cannot reach `llama-server` (it listens on 127.0.0.1 inside the inference worker and moves between
ports), so requests are queued in SQLite and answered by the inference worker through the same server
pool as Cut Assistance: this host's model or the remote llama.cpp PC, whichever is free. Chat has
priority over queued analysis jobs, and a running job is never interrupted.

## Enable it

1. Pick a key: `openssl rand -hex 24`.
2. Put `CHAT_API_KEY=<key>` in `.env` (empty keeps the endpoints disabled: they answer 503, because the
   rest of the API has no authentication).
3. Recreate the API container (`docker compose up -d`) and restart the inference worker
   (`docker restart video-analyzer-inference-worker-1`) so it starts the gateway thread.

Base URL: `http://<APP_BIND_IP>:<APP_PORT>/v1` (for this host `http://192.168.0.6:8080/v1`).

## Endpoints

- `GET /v1/models` lists one model, `qwen3.6-35b-a3b`.
- `POST /v1/chat/completions` takes `messages`, `stream`, `stream_options.include_usage`, `max_tokens` /
  `max_completion_tokens`, `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `stop`, penalties,
  `response_format`, `tools` and `tool_choice`. The `model` field is echoed back and does not choose a
  server.
- **Images:** a user message may mix `text` and `image_url` parts, as in OpenAI's API. The URL is either a
  base64 data URL (`data:image/png;base64,...`) or an `http(s)` URL that the API downloads (15 s, 20 MB).
  JPEG, PNG, GIF and BMP go to the model as they are; anything else ffmpeg can read (WebP, AVIF, TIFF, ...)
  is converted to PNG first. At most 16 images per request, only in `user` messages; `detail` is ignored.
  Such a request goes only to a server that can see: this host while its Qwen 3.6 (or Inkling) server has
  the vision projector loaded, or the remote PC when its chat model reports vision. When neither can, the
  answer is a 400 that says why for each server. Pictures count against the context (a 1024×576
  picture measured about 590 tokens on Qwen 3.6). The stored request drops the image data once it is answered.
- Thinking is on by default and arrives as `reasoning_content` (message and stream deltas). Turn it off
  with `"reasoning_effort": "none"` or `"chat_template_kwargs": {"enable_thinking": false}`.
- Errors use the OpenAI shape `{"error": {"message", "type", "code"}}`: 401 wrong key, 503 disabled or
  inference worker offline, 400 a request the model server rejects (for example over the context size),
  502 the server was unreachable.
- Closing the connection cancels the request.

```bash
curl http://192.168.0.6:8080/v1/chat/completions \
  -H "Authorization: Bearer $CHAT_API_KEY" -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

With a picture:

```bash
IMG=$(base64 -w0 photo.jpg)
curl http://192.168.0.6:8080/v1/chat/completions \
  -H "Authorization: Bearer $CHAT_API_KEY" -H 'Content-Type: application/json' \
  -d '{"reasoning_effort":"none","messages":[{"role":"user","content":[
        {"type":"text","text":"Describe this picture."},
        {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,'"$IMG"'"}}]}]}'
```

Any OpenAI client works with `base_url=http://192.168.0.6:8080/v1` and the key as API key.

## Limits

- The context per request is one server slot (`QWEN36_CONTEXT_SIZE / TEXT_JOB_CONCURRENCY`, 32768 by
  default) on this host, or the remote model's slot.
- Up to three gateway requests run at once (`chat_gateway_worker.MAX_CONCURRENT`); the rest wait. When both
  servers are busy a request waits with stage `Waiting for a free server`; while visual analysis holds the
  GPU and no remote PC is online it waits until the GPU is free.
- Plain HTTP on the LAN: anyone who can read the traffic can read the key.
