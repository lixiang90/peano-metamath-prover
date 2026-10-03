"""Read-only localhost monitor for a detached scale training session."""
import argparse
import ctypes
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import threading
import time


def process_alive(pid):
    if not pid:
        return False
    kernel = ctypes.windll.kernel32
    kernel.OpenProcess.restype = ctypes.c_void_p
    handle = kernel.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        ok = kernel.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code))
        return bool(ok and code.value == 259)
    finally:
        kernel.CloseHandle(ctypes.c_void_p(handle))


def read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_directory', type=Path)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    run = args.run_directory.resolve()
    gpu = {}
    history = []
    offset = 0
    lock = threading.Lock()
    def sampler():
        nonlocal gpu, offset
        while True:
            try:
                command = ['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw', '--format=csv,noheader,nounits']
                values = subprocess.check_output(command, text=True, timeout=5, creationflags=0x08000000).strip().splitlines()[0].split(', ')
                current = dict(zip(['name','utilization','memory_used','memory_total','temperature','power'], values))
                current['timestamp'] = time.time()
                with lock:
                    gpu = current
                with (run/'gpu.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(current)+'\n')
            except Exception as exc:
                with lock:
                    gpu = {'error':str(exc), 'timestamp':time.time()}
            time.sleep(5)
    threading.Thread(target=sampler,daemon=True).start()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *values):
            pass
        def do_GET(self):
            nonlocal offset
            route = self.path.split('?')[0]
            if route == '/':
                payload = Path(__file__).with_name('training-monitor.html').read_bytes()
                content_type = 'text/html; charset=utf-8'
            elif route == '/api/status':
                with lock:
                    path = run/'training.jsonl'
                    if path.exists():
                        with path.open('r',encoding='utf-8') as stream:
                            stream.seek(offset)
                            while True:
                                position = stream.tell()
                                line = stream.readline()
                                if not line:
                                    break
                                try:
                                    history.append(json.loads(line))
                                    offset = stream.tell()
                                except ValueError:
                                    offset = position
                                    break
                    # Aggregate contiguous steps to retain a full-run curve with
                    # bounded response size (the raw per-step log stays on disk).
                    stride = max(1,(len(history)+599)//600)
                    points = []
                    for start in range(0,len(history),stride):
                        chunk = history[start:start+stride]
                        points.append({key:sum(item[key] for item in chunk)/len(chunk) for key in ['step','loss','policy_loss','value_loss','learning_rate']})
                    gpu_copy = dict(gpu)
                state = read_json(run/'session.json')
                manifest = read_json(run/'corpus/manifest.json')
                checkpoint = run/'model/latest.pt'
                logs = []
                for name in ['prepare.log','runner.log','runner.err']:
                    p = run/name
                    if p.exists():
                        with p.open('rb') as stream:
                            stream.seek(max(0,p.stat().st_size-5000))
                            logs.extend(stream.read().decode('utf-8',errors='replace').splitlines()[-6:])
                result = {'now':time.time(),'session':state,'live':read_json(run/'live.json'),
                          'gpu':gpu_copy,'history':points,'runner_alive':process_alive(state.get('pid')),
                          'corpus':{k:manifest.get(k) for k in ['complete','counts','targets','proof_depth_histogram']},
                          'checkpoint':{'path':str(checkpoint),'modified':checkpoint.stat().st_mtime,'bytes':checkpoint.stat().st_size} if checkpoint.exists() else None,
                          'audit':read_json(run/'audit.json'),'logs':logs[-16:], 'run_directory':str(run)}
                payload = json.dumps(result,ensure_ascii=False).encode('utf-8')
                content_type = 'application/json; charset=utf-8'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type',content_type)
            self.send_header('Cache-Control','no-store')
            self.send_header('Content-Length',str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
    server = ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'Training monitor: http://127.0.0.1:{args.port}',flush=True)
    server.serve_forever()

if __name__ == '__main__':
    main()
