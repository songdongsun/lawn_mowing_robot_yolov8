"""LAN preview for a bounded standalone camera validation run."""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import os
import sys

os.chdir(sys.argv[1])
Path('index.html').write_text(r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>D02 ByteTrack</title>
<style>body{background:#151a20;color:#edf2f7;font:16px system-ui;max-width:1000px;margin:25px auto;padding:0 16px}
img{display:block;width:100%;max-width:960px}pre{white-space:pre-wrap}</style>
<h1>D02 KL + ByteTrack</h1><p>绿色：已确认轨迹；黄色：待确认。丢失轨迹不画成当前检测框。</p>
<p id="status">等待结果</p><img id="frame" alt="摄像头跟踪画面"><pre id="stats"></pre>
<script>
const frame=document.getElementById('frame');let last=-1,changed=performance.now();
function image(){frame.src='latest.jpg?t='+Date.now()}
frame.onload=()=>setTimeout(image,200);frame.onerror=()=>setTimeout(image,1000);image();
async function status(){try{
const r=await fetch('latest.json?t='+Date.now(),{cache:'no-store'});if(!r.ok)throw Error();
const d=await r.json();if(d.frame!==last){last=d.frame;changed=performance.now()}
const live=d.status==='running'&&performance.now()-changed<3000;
document.getElementById('status').textContent=live?(d.source==='video'?`视频推理中：第 ${d.loop} 遍，原视频帧 ${d.source_frame}/${d.source_frames}；每遍重新建立轨迹`:'摄像头跟踪中'):'已停止或无新帧，画面仅为最后一帧';
document.getElementById('stats').textContent=live?
`处理 ${d.fps.toFixed(1)} FPS | NPU ${d.inference_ms.toFixed(1)} ms | 跟踪 ${d.tracking_ms.toFixed(3)} ms\n`+
d.tracks.map(t=>`ID ${t.track_id} 类别 ${t.class_id} ${t.state} 本帧观测=${t.observed}`).join('\n'):d.status;
}catch(e){document.getElementById('status').textContent='尚无可用结果'}setTimeout(status,500)}status();
</script></html>''', encoding='utf-8')


class Preview(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()


ThreadingHTTPServer(('0.0.0.0', 8088), Preview).serve_forever()
