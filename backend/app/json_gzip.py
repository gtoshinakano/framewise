"""Gzip for complete JSON responses only.

Episode and transcript responses run to megabytes and shrink about tenfold. Video ranges, frames, audio and event
streams pass through untouched: compressing them wastes CPU, and a gzipped 206 breaks seeking. Starlette's own
GZipMiddleware only learned those exclusions in later releases, and the image does not pin Starlette.
"""
import gzip

from starlette.datastructures import Headers, MutableHeaders

MINIMUM_SIZE = 1024
LEVEL = 5  # ~6 ms for 1.5 MB; higher levels barely shrink JSON further


class JsonGzip:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or 'gzip' not in Headers(scope=scope).get('accept-encoding', ''):
            await self.app(scope, receive, send)
            return
        start, chunks = None, []

        async def respond(message):
            nonlocal start
            if message['type'] == 'http.response.start':
                headers = Headers(raw=message['headers'])
                if (message['status'] != 206 and 'content-encoding' not in headers
                        and headers.get('content-type', '').partition(';')[0].strip() == 'application/json'):
                    start = message
                    return
            if start is None or message['type'] != 'http.response.body':
                await send(message)
                return
            chunks.append(message.get('body', b''))
            if message.get('more_body'):
                return
            body = b''.join(chunks)
            headers = MutableHeaders(raw=start['headers'])
            if len(body) >= MINIMUM_SIZE:
                body = gzip.compress(body, LEVEL)
                headers['content-encoding'] = 'gzip'
                headers['content-length'] = str(len(body))
                headers.add_vary_header('Accept-Encoding')
            await send(start)
            await send({'type': 'http.response.body', 'body': body})

        await self.app(scope, receive, respond)
