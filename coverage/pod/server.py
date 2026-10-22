#!/usr/bin/env python3
"""Tiny file and job server for a throwaway GPU pod, reached through the RunPod HTTP proxy.

Every request needs the header X-Token: $POD_TOKEN.
    PUT  /f/<path>      write a file under /work
    GET  /f/<path>      read a file under /work
    POST /run           body is a shell command; runs detached, output in /work/logs/<n>.log
    GET  /ls/<path>     list a directory under /work
"""
import itertools
import json
import os
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path('/work')
TOKEN = os.environ['POD_TOKEN']
jobs = itertools.count(1)


class Handler(BaseHTTPRequestHandler):
    def _path(self, prefix):
        p = (ROOT / self.path[len(prefix):].lstrip('/')).resolve()
        if not str(p).startswith(str(ROOT)):
            raise PermissionError(p)
        return p

    def _ok(self, body=b'', code=200, kind='application/octet-stream'):
        self.send_response(code)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _auth(self):
        if self.headers.get('X-Token') != TOKEN:
            self._ok(b'no', 403)
            return False
        return True

    def do_GET(self):
        if not self._auth():
            return
        try:
            if self.path.startswith('/ls/') or self.path == '/ls':
                d = self._path('/ls')
                self._ok(json.dumps({p.name: p.stat().st_size for p in sorted(d.iterdir())}).encode(), kind='application/json')
            else:
                self._ok(self._path('/f').read_bytes())
        except Exception as e:  # noqa: BLE001 - report anything to the caller
            self._ok(str(e).encode(), 404)

    def do_PUT(self):
        if not self._auth():
            return
        p = self._path('/f')
        p.parent.mkdir(parents=True, exist_ok=True)
        n = int(self.headers['Content-Length'])
        with open(p, 'wb') as f:
            while n:
                chunk = self.rfile.read(min(n, 1 << 20))
                f.write(chunk)
                n -= len(chunk)
        self._ok(b'ok')

    def do_POST(self):
        if not self._auth():
            return
        cmd = self.rfile.read(int(self.headers['Content-Length'])).decode()
        n = next(jobs)
        (ROOT / 'logs').mkdir(parents=True, exist_ok=True)
        log = open(ROOT / 'logs' / f'{n}.log', 'wb')
        subprocess.Popen(['bash', '-c', f'{cmd}; echo "== exit $?"'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self._ok(json.dumps({'job': n, 'log': f'logs/{n}.log'}).encode(), kind='application/json')

    def log_message(self, *a):
        pass


if __name__ == '__main__':
    ROOT.mkdir(exist_ok=True)
    print('listening on 0.0.0.0:8000', flush=True)
    ThreadingHTTPServer(('0.0.0.0', 8000), Handler).serve_forever()
