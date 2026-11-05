"""Stage/run/fetch the independent board audit. Read SSH password from stdin."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import paramiko

OUT=Path(__file__).resolve().parents[1]
REMOTE='/root/d02_detect_static_20261103'
variant=sys.argv[2] if len(sys.argv)>2 else 'uint8_hybrid'
assert variant in ['uint8_hybrid','fp16','head_tail_fp16','attention_tail_fp16','attention_tail_kl','p5_class_fp16','coco80_kl','coco80_fp16','coco80_int8','delivery_D02_KL']
local_board=OUT/'board'
if variant!='uint8_hybrid':
    REMOTE+='/'+variant
    local_board=local_board/variant
local_board.mkdir(exist_ok=True)
client=paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('192.168.8.11',username='root',password=sys.stdin.readline().strip(),look_for_keys=False,allow_agent=False,timeout=10)
records=[]
def run(command):
    _,out,err=client.exec_command(command,timeout=60)
    a=out.read().decode();b=err.read().decode();rc=out.channel.recv_exit_status()
    records.append(dict(command=command,stdout=a,stderr=b,exit_code=rc))
    print(a,b,flush=True)
    if rc: raise RuntimeError(f'Remote command failed: {rc}')
    return a
mode=sys.argv[1]
try:
    if mode=='stage' and variant!='uint8_hybrid':
        run('test ! -e '+REMOTE+' && mkdir '+REMOTE+' && cd '+REMOTE+' && ln -s ../inputs inputs && cp ../input_list.txt ../detectStatic .')
    elif mode=='stage':
        payload=OUT/'board/payload.tar'
        with tarfile.open(payload,'w') as tar:
            for name in ['inputs','input_list.txt','detectStatic']:
                tar.add(OUT/'board'/name,arcname=name)
        run('test ! -e '+REMOTE+' && mkdir '+REMOTE)
        with client.open_sftp() as sftp:sftp.put(str(payload),REMOTE+'/payload.tar')
        digest=hashlib.sha256(payload.read_bytes()).hexdigest()
        got=run('sha256sum '+REMOTE+'/payload.tar').split()[0];assert got==digest
        run('cd '+REMOTE+' && tar xf payload.tar && chmod +x detectStatic')
    elif mode=='run':
        if variant in ['coco80_kl','coco80_fp16','coco80_int8','delivery_D02_KL']:
            with client.open_sftp() as sftp:sftp.put(str(OUT/'board/detectStatic'),REMOTE+'/detectStatic')
            run('chmod +x '+REMOTE+'/detectStatic')
        model=(OUT/'delivery_D02_KL' if variant=='delivery_D02_KL' else OUT/'variants'/variant)/'network_binary.nb'
        with client.open_sftp() as sftp:sftp.put(str(model),REMOTE+'/network_binary.nb')
        got=run('sha256sum '+REMOTE+'/network_binary.nb').split()[0]
        assert got==hashlib.sha256(model.read_bytes()).hexdigest()
        run('cd '+REMOTE+' && mkdir outputs && (nohup sh -c \'LD_LIBRARY_PATH=/root/yolov8n_detect_npu_20260706/lib:$LD_LIBRARY_PATH ./detectStatic > board_run.log 2>&1; echo $? > exit_code.txt\' > launcher.log 2>&1 < /dev/null & )')
    elif mode=='status':
        run('cd '+REMOTE+' && tail -8 board_run.log; if test -f '+REMOTE+'/exit_code.txt; then cat '+REMOTE+'/exit_code.txt; fi')
    elif mode=='fetch':
        run('cd '+REMOTE+' && test "$(cat exit_code.txt)" = 0')
        _,out,err=client.exec_command('cd '+REMOTE+' && sha256sum outputs/* board_run.log exit_code.txt',timeout=60)
        hashes=out.read().decode();errors=err.read().decode()
        if out.channel.recv_exit_status():raise RuntimeError(errors)
        (local_board/'remote_output_sha256.txt').write_text(hashes)
        dst=local_board/'results.tar'
        _,out,err=client.exec_command('cd '+REMOTE+' && tar cf - outputs board_run.log exit_code.txt',timeout=60)
        with dst.open('wb') as stream:
            while True:
                chunk=out.read(1024*1024)
                if not chunk:break
                stream.write(chunk)
        errors=err.read().decode()
        if out.channel.recv_exit_status():raise RuntimeError(errors)
        with tarfile.open(dst) as tar:
            for m in tar.getmembers():
                assert not Path(m.name).is_absolute() and '..' not in Path(m.name).parts and (m.isfile() or m.isdir())
            tar.extractall(local_board)
        for row in hashes.splitlines():
            digest,name=row.split('  ',1)
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            assert hashlib.sha256((local_board/name).read_bytes()).hexdigest()==digest,name
        records.append(dict(mode='stream_tar',verified_files=len(hashes.splitlines())))
        print('Fetched and verified',len(hashes.splitlines()),'files',flush=True)
    else:raise ValueError(mode)
finally:
    (local_board/f'{mode}_commands.json').write_text(json.dumps(records,indent=2))
    client.close()
