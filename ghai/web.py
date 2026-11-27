"""GHAi web arayüzü — tarayıcıdan sohbet.

Kullanım:
  python -m ghai.web --model dist/GHAi-3B-Q4_K_M.gguf
  Ardından tarayıcıda http://localhost:7860 adresini açın.

Ek bağımlılık gerektirmez (yalnızca llama-cpp-python).
"""
import argparse, json, os, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .engine import GHAiEngine, DEFAULT_MODEL

PAGE = r"""<!doctype html>
<html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GHAi</title>
<style>
:root{--bg:#0f1115;--panel:#171a21;--line:#262a33;--text:#e7e9ee;--muted:#8b93a3;--acc:#3b82f6;--user:#1e3a5f}
@media (prefers-color-scheme: light){:root{--bg:#f6f7f9;--panel:#fff;--line:#e3e6eb;--text:#151821;--muted:#667085;--acc:#2563eb;--user:#dbeafe}}
*{box-sizing:border-box}html,body{height:100%;margin:0}
body{background:var(--bg);color:var(--text);font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;display:flex;flex-direction:column}
header{padding:14px 16px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:10px}
.logo{width:34px;height:34px;border-radius:9px;background:linear-gradient(135deg,#2563eb,#7c3aed);display:grid;place-items:center;font-weight:700;color:#fff;font-size:13px}
header b{font-size:17px}header small{display:block;color:var(--muted);font-size:12px}
header button{margin-left:auto;background:none;border:1px solid var(--line);color:var(--text);border-radius:8px;padding:6px 12px;cursor:pointer}
#log{flex:1;overflow-y:auto;padding:20px 16px}
.wrap{max-width:780px;margin:0 auto}
.msg{margin:0 0 16px;display:flex}.msg div{padding:10px 14px;border-radius:14px;max-width:88%;white-space:pre-wrap;word-wrap:break-word}
.user{justify-content:flex-end}.user div{background:var(--user)}
.bot div{background:var(--panel);border:1px solid var(--line)}
.empty{text-align:center;color:var(--muted);margin-top:18vh}.empty h1{color:var(--text);margin:0 0 6px;font-size:28px}
form{border-top:1px solid var(--line);padding:12px 16px}
.row{max-width:780px;margin:0 auto;display:flex;gap:8px}
textarea{flex:1;resize:none;background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:12px;padding:11px 13px;font:inherit;min-height:46px;max-height:180px}
textarea:focus{outline:2px solid var(--acc);border-color:transparent}
#send{background:var(--acc);color:#fff;border:0;border-radius:12px;padding:0 18px;font-weight:600;cursor:pointer}
#send:disabled{opacity:.5}
code{background:rgba(127,127,127,.15);padding:1px 5px;border-radius:5px}
</style></head><body>
<header><div class="logo">GH</div><div><b>GHAi</b><small>Gökalp Hizmetleri</small></div><button id="new">Yeni sohbet</button></header>
<div id="log"><div class="wrap" id="msgs"><div class="empty"><h1>GHAi</h1>Size nasıl yardımcı olabilirim?</div></div></div>
<form id="f"><div class="row"><textarea id="q" rows="1" placeholder="Mesajınızı yazın..." autofocus></textarea><button id="send">Gönder</button></div></form>
<script>
const msgs=document.getElementById('msgs'),q=document.getElementById('q'),send=document.getElementById('send'),log=document.getElementById('log');
let history=[];
function add(role,text){const e=msgs.querySelector('.empty');if(e)e.remove();
 const m=document.createElement('div');m.className='msg '+role;const d=document.createElement('div');d.textContent=text;m.appendChild(d);msgs.appendChild(m);log.scrollTop=log.scrollHeight;return d}
document.getElementById('new').onclick=()=>{history=[];msgs.innerHTML='<div class="empty"><h1>GHAi</h1>Size nasıl yardımcı olabilirim?</div>'};
q.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();document.getElementById('f').requestSubmit()}});
q.addEventListener('input',()=>{q.style.height='auto';q.style.height=q.scrollHeight+'px'});
document.getElementById('f').onsubmit=async e=>{e.preventDefault();const text=q.value.trim();if(!text||send.disabled)return;
 q.value='';q.style.height='auto';add('user',text);history.push({role:'user',content:text});
 const d=add('bot','…');send.disabled=true;let out='';
 try{const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({messages:history})});
  const rd=r.body.getReader(),dec=new TextDecoder();
  while(true){const {value,done}=await rd.read();if(done)break;out+=dec.decode(value,{stream:true});d.textContent=out;log.scrollTop=log.scrollHeight}
 }catch(err){out='Bir hata oluştu: '+err;d.textContent=out}
 history.push({role:'assistant',content:out});send.disabled=false;q.focus()};
</script></body></html>"""


def make_handler(engine, lock, system):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != "/api/chat":
                self.send_error(404); return
            n = int(self.headers.get("Content-Length", 0))
            try:
                msgs = json.loads(self.rfile.read(n))["messages"]
                msgs = [{"role": m["role"], "content": str(m["content"])} for m in msgs
                        if m.get("role") in ("user", "assistant")]
            except Exception:
                self.send_error(400); return
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            with lock:
                try:
                    for piece in engine.stream(msgs, system=system):
                        b = piece.encode()
                        self.wfile.write(b"%x\r\n%s\r\n" % (len(b), b))
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
            self.wfile.write(b"0\r\n\r\n")

    return H


def main():
    ap = argparse.ArgumentParser(description="GHAi web arayüzü")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--system", default=None)
    a = ap.parse_args()
    if not os.path.exists(a.model):
        sys.exit(f"Model bulunamadı: {a.model}")
    print("GHAi yükleniyor...", flush=True)
    eng = GHAiEngine(a.model)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(eng, threading.Lock(), a.system))
    print(f"GHAi hazır: http://{a.host}:{a.port}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
