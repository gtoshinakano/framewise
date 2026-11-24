"""Read-only search over Framewise's configured public sources."""
import http.client
import ipaddress
import json
import re
import socket
import ssl
import time
from html.parser import HTMLParser
from urllib.parse import quote, unquote, urlencode, urljoin, urlsplit, urlunsplit

from . import db

DEFAULTS = [
    {'kind': 'wiki', 'url': 'https://ja.wikipedia.org/w/api.php'},
    {'kind': 'wiki', 'url': 'https://en.wikipedia.org/w/api.php'},
]
MAX_BYTES = 2 * 1024 * 1024
MAX_HITS = 6
SKIP = {'script', 'style', 'svg', 'noscript', 'footer', 'nav', 'head', 'rp'}
BLOCK = {'p', 'li', 'div', 'tr', 'td', 'th', 'dt', 'dd', 'section', 'br', 'h1', 'h2', 'h3'}


def normalize(url):
    url = url.strip()
    parts = urlsplit(url)
    if (parts.scheme != 'https' or not parts.hostname or parts.username is not None or parts.password is not None
            or parts.port not in (None, 443) or any(c.isspace() or ord(c) < 32 for c in url)):
        raise ValueError('Enter a public HTTPS URL.')
    host = parts.hostname.encode('idna').decode('ascii').lower().rstrip('.')
    if not re.fullmatch(r'[a-z0-9-]+(?:\.[a-z0-9-]+)+', host):
        raise ValueError('Enter a public website hostname.')
    path = unquote(parts.path or '/', errors='strict')
    if any(c in path for c in ('\\', '%')) or any(ord(c) < 32 for c in path) or any(p in ('.', '..') for p in path.split('/')):
        raise ValueError('Enter a URL with a canonical path.')
    return urlunsplit(('https', host, quote(path, safe="/:@!$&'()*+,;=-._~"), quote(parts.query, safe="=&;%:+?,@/!$'()*-._~"), ''))


def clean_source(kind, url):
    url = normalize(url)
    parts = urlsplit(url)
    if kind == 'wiki':
        path = parts.path
        if not path.endswith('/api.php'):
            if not path.rstrip('/'):
                path = '/w/api.php'
            else:
                path = path.rstrip('/') + '/api.php'
        url = parts._replace(path=path, query='').geturl()
    return {'kind': kind, 'url': url}


def sources():
    row = db.one('SELECT sources_json FROM web_search_settings WHERE id=1')
    return json.loads(row['sources_json']) if row else DEFAULTS


def allowed(source, url):
    base, target = urlsplit(source['url']), urlsplit(normalize(url))
    if base.netloc != target.netloc:
        return False
    if source['kind'] == 'page':
        return (base.path, base.query) == (target.path, target.query)
    if source['kind'] == 'wiki':
        return True
    prefix = base.path.rstrip('/')
    return target.path == prefix or target.path.startswith(prefix + '/')


def public_addresses(host):
    addresses = list(dict.fromkeys(item[4][0] for item in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)))
    if not addresses:
        raise ValueError('Could not resolve the source.')
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global or ip.is_multicast or getattr(ip, 'ipv4_mapped', None):
            raise ValueError('The source must resolve to public addresses.')
    return addresses


class PublicHTTPS(http.client.HTTPSConnection):
    def connect(self):
        # Connect to the checked address while verifying TLS against the website hostname.
        addresses = public_addresses(self.host)
        for address in addresses[:3]:
            try:
                sock = socket.create_connection((address, self.port), self.timeout)
            except OSError:
                continue
            try:
                self.sock = self._context.wrap_socket(sock, server_hostname=self.host)
                return
            except BaseException:
                sock.close()
                raise
        raise OSError('Could not reach the source.')


def fetch(source, url, deadline):
    for _ in range(4):
        url = normalize(url)
        if not allowed(source, url):
            raise ValueError('The destination is outside the approved source.')
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Source search timed out.')
        parts = urlsplit(url)
        connection = PublicHTTPS(parts.hostname, timeout=min(4, remaining), context=ssl.create_default_context())
        try:
            connection.request('GET', parts.path + ('?' + parts.query if parts.query else ''), headers={
                'User-Agent': 'Framewise/1.0 (user-initiated character and source lookup)',
                'Accept': 'text/html, application/json, text/plain',
            })
            response = connection.getresponse()
            if response.status in (301, 302, 303, 307, 308):
                location = response.getheader('Location')
                if not location:
                    raise ValueError('Source redirect has no destination.')
                url = urljoin(url, location)
                continue
            if response.status != 200:
                raise ValueError(f'Source returned HTTP {response.status}.')
            content_type = response.headers.get_content_type()
            if content_type not in ('text/html', 'application/json', 'text/plain', 'application/xhtml+xml'):
                raise ValueError('The source did not return a text page.')
            chunks = bytearray()
            while len(chunks) <= MAX_BYTES and time.monotonic() < deadline:
                chunk = response.read1(min(65536, MAX_BYTES + 1 - len(chunks)))
                if not chunk:
                    return url, chunks.decode(response.headers.get_content_charset() or 'utf-8', errors='replace'), content_type
                chunks.extend(chunk)
            raise ValueError('Source response exceeded the search limits.')
        finally:
            connection.close()
    raise ValueError('Too many source redirects.')


class Page(HTMLParser):
    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.fragments, self.links, self.title = [], [], []
        self.skip = []
        self.anchor = None
        self.in_title = False
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        if tag == 'title':
            self.in_title = True
        if tag in SKIP:
            self.skip.append(tag)
        if self.skip:
            return
        if tag in BLOCK:
            self.fragments.append('\n')
        if tag == 'rt':
            self.fragments.append('（')
        if tag == 'a':
            self.anchor = (dict(attrs).get('href'), [])

    def handle_endtag(self, tag):
        if tag == 'title':
            self.in_title = False
        if self.skip:
            if tag == self.skip[-1]:
                self.skip.pop()
            return
        if tag in BLOCK:
            self.fragments.append('\n')
        if tag == 'rt':
            self.fragments.append('）')
        if tag == 'a' and self.anchor:
            url, label = self.anchor
            if url:
                self.links.append((url, ''.join(label).strip()))
            self.anchor = None

    def handle_data(self, data):
        if self.in_title:
            self.title.append(data)
        if not self.skip:
            self.fragments.append(data)
            if self.anchor:
                self.anchor[1].append(data)

    def excerpt(self, query):
        lines = [' '.join(line.split()) for line in ''.join(self.fragments).splitlines() if line.strip()]
        subject = name_text(query)
        terms = keywords(query)
        matches = []
        for i, line in enumerate(lines):
            if re.match(r'^\d+(?:\.\d+)*\s', line):
                continue
            folded = name_text(line)
            score = sum(term in folded for term in terms) + (len(terms) + 1 if subject and subject in folded else 0)
            if score:
                matches.append((score, i))

        # Put the evidence first so it survives the smaller excerpt budget for model requests.
        selected, included = [], set()
        for _, i in sorted(matches, key=lambda match: (-match[0], match[1])):
            for index in (i, i - 1, i + 1, i + 2):
                if 0 <= index < len(lines) and index not in included:
                    selected.append(lines[index][:800])
                    included.add(index)
            if sum(len(line) for line in selected) >= 4600:
                break
        return '\n'.join(selected or lines)[:4600]


def name_text(text):
    text = re.sub(r'[（(][ぁ-ゖァ-ヺー 　]+[）)]', '', text.casefold())
    return ''.join(text.split())


def keywords(query):
    terms = re.findall(r'[^\W_]+', query.casefold())
    return [term for term in terms if len(term) >= 2]


def page_hit(source, url, query, deadline):
    url, body, kind = fetch(source, url, deadline)
    if kind == 'application/json':
        raise ValueError('Choose a page containing readable text.')
    page = Page(body)
    return {'url': url, 'title': ''.join(page.title).strip()[:240] or url, 'excerpt': page.excerpt(query)}, page


def wiki_hits(source, query, subject, deadline):
    endpoint = source['url']

    def api(params):
        return json.loads(fetch(source, endpoint + '?' + urlencode({'format': 'json', 'formatversion': 2, **params}), deadline)[1])

    def find(terms):
        rows = api({'action': 'query', 'generator': 'search', 'gsrsearch': terms, 'gsrlimit': 4,
                    'prop': 'info', 'inprop': 'url'}).get('query', {}).get('pages', [])
        return sorted(rows, key=lambda row: row.get('index', 0))

    try:
        result = find(subject or query)
    except ValueError as exc:
        if urlsplit(endpoint).path != '/w/api.php' or ('HTTP 404' not in str(exc) and not isinstance(exc, json.JSONDecodeError)):
            raise
        endpoint = urljoin(endpoint, '/api.php')
        result = find(subject or query)
    if subject and subject != query:
        candidates = {row['pageid']: row for row in [*result, *find(query)]}
        normalized = name_text(subject)
        result = sorted(candidates.values(), key=lambda row: (
            normalized not in name_text(row['title']),
            not re.search(r'登場人物|characters', row['title'], re.IGNORECASE)))
    for row in result[:2]:
        link = row.get('fullurl')
        if not link or not allowed(source, link):
            continue
        parsed = api({'action': 'parse', 'pageid': row['pageid'], 'prop': 'text', 'disableeditsection': 1}).get('parse', {})
        if 'text' in parsed:
            page = Page(parsed['text'])
            yield {'title': parsed.get('title', row['title']), 'url': normalize(link), 'excerpt': page.excerpt(subject or query)}


def search(query, *, subject=None, selected_url=None):
    """Return citation-ready text. Provider adapters can call this before any LLM request."""
    configured = sources()
    deadline = time.monotonic() + 15
    hits, warnings, seen = [], [], set()
    for source in configured:
        if len(hits) >= MAX_HITS or time.monotonic() >= deadline:
            break
        try:
            if selected_url:
                if not allowed(source, selected_url):
                    continue
                candidates = [page_hit(source, selected_url, subject or query, deadline)[0]]
            elif source['kind'] == 'wiki':
                candidates = wiki_hits(source, query, subject, deadline)
            else:
                hit, page = page_hit(source, source['url'], subject or query, deadline)
                candidates = [hit]
                if source['kind'] == 'directory':
                    terms = keywords(subject or query)
                    links = []
                    for href, label in page.links:
                        if not any(term in (label + ' ' + unquote(href)).casefold() for term in terms):
                            continue
                        try:
                            link = normalize(urljoin(hit['url'], href))
                            if link != hit['url'] and link not in links and allowed(source, link):
                                links.append(link)
                        except (ValueError, UnicodeError):
                            continue
                    for link in links[:2]:
                        try:
                            candidates.append(page_hit(source, link, subject or query, deadline)[0])
                        except (ValueError, OSError, http.client.HTTPException, LookupError):
                            continue
            for hit in candidates:
                if hit['excerpt'] and hit['url'] not in seen:
                    seen.add(hit['url'])
                    hits.append({'id': len(hits) + 1, **hit})
                if len(hits) == MAX_HITS:
                    break
        except (ValueError, OSError, http.client.HTTPException, LookupError) as exc:
            warnings.append({'source': source['url'], 'message': str(exc)[:160]})
        if selected_url and hits:
            break
    return {'hits': hits, 'warnings': warnings, 'configured': bool(configured)}
