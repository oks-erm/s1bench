import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from http.server import ThreadingHTTPServer

import report_server as s


class ReportServerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        self.payload = {'baseline':'gpt','cases':[{'id':'one'},{'id':'two'}],
                        'summaries_by_reference':{'gpt':[{'model':'jev','dataset':'evaluation','success_rate':.8},
                                                        {'model':'jev','dataset':'development','success_rate':1}]}}
        self.html = ("<!doctype html><html lang='en'><body><script id='report-data' type='application/json'>"
                     + json.dumps(self.payload) + "</script></body></html>").encode()
        (self.folder/'report.html').write_bytes(self.html)
        (self.folder/'config.json').write_text('private')

    def tearDown(self):
        self.tmp.cleanup()

    def test_downloads_preserve_complete_html_and_data_and_filter_csv_only(self):
        self.assertEqual(s.download_content(self.folder,'report.html')[0],self.html)
        data = s.download_content(self.folder,'data.json')[0]
        self.assertEqual(json.loads(data),self.payload)
        csv = s.download_content(self.folder,'comparison.csv',baseline='gpt',dataset='evaluation')[0].decode('utf-8-sig')
        self.assertIn('evaluation',csv)
        self.assertNotIn('development',csv)
        self.assertEqual((self.folder/'report.html').read_bytes(),self.html)

    def test_rejects_unknown_downloads_and_references(self):
        with self.assertRaises(ValueError):s.download_content(self.folder,'config.json')
        with self.assertRaises(ValueError):s.download_content(self.folder,'comparison.csv',baseline='unknown')

    def test_http_attachments_and_allowlist(self):
        server = ThreadingHTTPServer(('127.0.0.1',0),s.handler_for(self.folder))
        thread = threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base = 'http://127.0.0.1:'+str(server.server_port)
        try:
            with urllib.request.urlopen(base+'/download/report.html') as response:
                self.assertIn('attachment',response.headers['Content-Disposition'])
                self.assertEqual(response.read(),self.html)
            with urllib.request.urlopen(base+'/report.html') as response:
                self.assertIn(b'data-downloads="http"',response.read())
            for path in ('/config.json','/../config.json','/download/config.json'):
                with self.assertRaises(urllib.error.HTTPError) as error:urllib.request.urlopen(base+path)
                self.assertEqual(error.exception.code,404)
        finally:
            server.shutdown();thread.join();server.server_close()
