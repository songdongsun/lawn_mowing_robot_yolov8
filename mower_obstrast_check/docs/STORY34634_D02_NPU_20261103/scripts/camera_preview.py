"""Serve only the bounded camera preview directory on the board LAN."""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import os
import sys

os.chdir(sys.argv[1])
Path('index.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>D02 摄像头检测</title>
<style>body{background:#151a20;color:#edf2f7;font:16px system-ui;max-width:1000px;margin:25px auto;padding:0 16px}
img{display:block;width:100%;max-width:960px;border-radius:8px}#state{padding:12px 0;color:#b4d7e8}
.muted{color:#a5afba}#stats{margin:14px 0}</style>
<h1>D02 障碍物检测</h1><div class="muted">八类 KL 模型 · USB 摄像头 · 预览约 5 帧/秒</div>
<div id="state">正在连接摄像头…</div><img id="frame" alt="摄像头检测画面"><div id="stats"></div>
<div class="muted">人员、猫、狗、飞碟、运动球、玩具熊、软管、石头。置信度 0.25。</div>
<script>
const frame=document.getElementById('frame'),state=document.getElementById('state');
let last=-1,changed=performance.now();
function image(){frame.src='latest.jpg?t='+Date.now()}
frame.onload=()=>setTimeout(image,200);frame.onerror=()=>setTimeout(image,1000);image();
async function status(){try{const r=await fetch('latest.json?t='+Date.now(),{cache:'no-store'});
if(!r.ok)throw Error();const d=await r.json();if(d.frame!==last){last=d.frame;changed=performance.now()}
const live=performance.now()-changed<3000;state.textContent=live?'摄像头推理中':'画面已停止更新，请检查摄像头进程';
state.style.color=live?'#87e5a3':'#ffb57b';document.getElementById('stats').textContent=
`处理帧率 ${d.fps.toFixed(1)} FPS · NPU ${d.inference_ms.toFixed(1)} ms · 检测 ${d.boxes.length} 个 · 帧号 ${d.frame}`;
}catch(e){state.textContent='暂未收到检测结果，请检查摄像头连接'}setTimeout(status,1000)}status();
</script></html>''',encoding='utf-8')

class Preview(SimpleHTTPRequestHandler):
    def log_message(self,*args):
        pass

    def end_headers(self):
        self.send_header('Cache-Control','no-store')
        super().end_headers()

ThreadingHTTPServer(('0.0.0.0',8088),Preview).serve_forever()
