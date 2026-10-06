"""Serve one saved report and HTTP attachment downloads on localhost only."""
from __future__ import annotations

import argparse
import csv
import io
import json
from html.parser import HTMLParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


class PayloadParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.active = False
        self.parts = []
        self.count = 0

    def handle_starttag(self, tag, attrs):
        if tag == 'script' and dict(attrs).get('id') == 'report-data':
            self.active = True
            self.count += 1

    def handle_endtag(self, tag):
        if tag == 'script':
            self.active = False

    def handle_data(self, data):
        if self.active:
            self.parts.append(data)


def saved_payload(html):
    parser = PayloadParser()
    parser.feed(html.decode('utf-8'))
    if parser.count != 1:
        raise ValueError('A complete saved report is required.')
    return json.loads(''.join(parser.parts))


def download_content(folder, kind, *, baseline=None, dataset=''):
    """Read saved exports only; never execute models or expose other files."""
    if kind not in {'report.html', 'data.json', 'comparison.csv'}:
        raise ValueError('Unknown download.')
    html = (Path(folder)/'report.html').read_bytes()
    if kind == 'report.html':
        return html, 'text/html; charset=utf-8', 'benchmark_report.html'
    payload = saved_payload(html)
    if kind == 'data.json':
        return json.dumps(payload,ensure_ascii=False,allow_nan=False).encode(), 'application/json', 'benchmark_report_data.json'
    references = payload['summaries_by_reference']
    baseline = baseline or payload['baseline']
    if baseline not in references:
        raise ValueError('Unknown reference.')
    rows = [r for r in references[baseline] if not dataset or r['dataset'] == dataset]
    fields = list(dict.fromkeys(k for row in rows for k in row))
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream,fieldnames=fields)
    writer.writeheader()
    writer.writerows({k: json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v
                     for k,v in row.items()} for row in rows)
    return stream.getvalue().encode('utf-8-sig'), 'text/csv; charset=utf-8', 'benchmark_comparison.csv'


def handler_for(folder):
    folder = Path(folder).resolve()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.respond(False)

        def do_HEAD(self):
            self.respond(True)

        def respond(self, head):
            request = urlsplit(self.path)
            attachment = request.path.startswith('/download/')
            try:
                if request.path in {'/', '/report.html'}:
                    body = (folder/'report.html').read_bytes().replace(
                        b"<html lang='en'>", b'<html lang="en" data-downloads="http">', 1)
                    mime, name = 'text/html; charset=utf-8', ''
                elif request.path in {'/download/report.html','/download/data.json','/download/comparison.csv'}:
                    query = parse_qs(request.query)
                    body,mime,name = download_content(folder,request.path.rsplit('/',1)[-1],
                        baseline=query.get('baseline',[None])[0],dataset=query.get('dataset',[''])[0])
                else:
                    self.send_error(404,'Unknown report file.')
                    return
            except (OSError,ValueError,KeyError):
                self.send_error(400,'Saved report download unavailable.')
                return
            self.send_response(200)
            self.send_header('Content-Type',mime)
            self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            if attachment:
                self.send_header('Content-Disposition','attachment; filename="'+name+'"')
            self.end_headers()
            if not head:
                self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    parser.add_argument('--port',type=int,default=8767)
    args = parser.parse_args()
    if not (args.folder/'report.html').is_file():
        parser.error('The folder must contain report.html.')
    server = ThreadingHTTPServer(('127.0.0.1',args.port),handler_for(args.folder))
    print(f'Report: http://127.0.0.1:{server.server_port}/report.html',flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
