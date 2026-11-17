# Remote llama.cpp text server

Framewise can send local-provider text jobs to a llama.cpp server on another PC on the LAN. These are consistency summaries, production decisions, Cut summaries and sequence steps. Cut Assistance can also use it. The worker treats the remote as extra lanes on the same queue. A job goes to whichever lane is free, as the routing table in the Text engine dialog allows. Remote lanes never wait for this host's GPU, so text work continues while visual analysis runs here, also for the same episode. A job a remote lane runs does not hold its episode against the next visual analysis job either. Other work on the episode (transcription, subtitle extraction, character detection, media preparation, hard-cut detection) still runs alone: a remote text job waits for it, and it waits for a remote text job.

## What Framewise expects from the other PC

- **One base URL** (for example `http://192.168.0.20:8080`) with the OpenAI-compatible endpoints `/v1/chat/completions` and `/v1/models`. `/health` is optional; a router that answers it with 404 is fine.
- **The `model` field picks the model.** Serve Qwen 3.6 MTP and Bonsai2 behind the same port with llama.cpp's router mode (`llama-server` started with a models directory or preset instead of `-m`) or with llama-swap. Running two `llama-server` processes on two ports is not supported by one entry. Framewise lists whatever `/v1/models` reports and lets you route each job type to one of those ids.
- **Per-slot context.** Framewise reads `default_generation_settings.n_ctx` from `/props?model=<id>` to size sequence inputs. When it cannot, it assumes 32768.
- **Flags that match this host's server.** Use `--jinja` (the Qwen template and `enable_thinking`), `--no-context-shift`, `--reasoning-budget 2048`, and `--parallel N` where N is the "Parallel jobs" value in the dialog. For the MTP model add `--spec-type draft-mtp --spec-draft-n-max 3`.
- **Access token (optional).** Start `llama-server` with `--api-key <token>`. Enter the same token in the dialog; Framewise sends `Authorization: Bearer <token>` with every request (it is stored in the database, never shown again, and can be replaced or removed there). Long generations can take many minutes, so do not give the proxy a short read timeout. Framewise waits up to 20 minutes per request.

## Dynamic IP

The other PC has no host name, so Framewise uses the address typed in the dialog. When the PC gets a new IP, enter it again and save. Alternatively, give the PC a DHCP reservation on the router. While the address is wrong, the dialog shows the server as offline:

- Jobs routed "Remote only" wait in the queue.
- Jobs routed "or this server" run here.
- A job whose server stops answering mid-generation goes back to the queue, and its attempt is not counted.

## Checking it

1. Text engine dialog → Remote llama.cpp server: enter the address and the access token, then press **Test connection**. Both models should be listed with their context.
2. Tick **Use for text jobs**, set the routing and save. Within about 10 s the status line shows *Online* and the header button shows `· Remote online`.
3. Queued results record `usage.model_variant = "remote:<model id>"`. The speed table in the dialog shows a row per remote model next to the local variants.

## One pool for chat, summaries and sequences

Both servers (this host and the remote PC) are treated as slots in one pool. Unless you chose a route for a job type, Consistency summaries, Production decisions, Cut summaries and Sequence steps go to whichever server is free first, using the remote model named under "Remote model for chat and shared jobs" (default: the first model the server lists). "Share every job type with the remote" in the dialog fills all four routes at once.

- **Chat first.** Cut Assistance has an **Auto · first free server** provider (the default once the remote is online). It answers on this host or the remote, whichever has a free slot, preferring the one chosen under "When both servers are free, prefer". While a chat message waits, no new text job starts, and each server keeps **Slots kept free for chat** (default 1) unavailable to jobs, so a reply never waits behind a running job. Running jobs are not interrupted.
- **Slots.** This host has `TEXT_JOB_CONCURRENCY` slots (default 2; restart the inference worker and its llama-server after changing it), the remote has "Parallel jobs". With a reserve of 1 and 2 local slots only one job runs here at a time, so set `TEXT_JOB_CONCURRENCY=3` to keep two.
- **Saved settings.** Address, token, parallel jobs, routing, preferred server, chat reserve and the remote chat model are saved in the database and survive restarts; change them in the Text engine dialog (no restart needed, the workers read them every few seconds).
