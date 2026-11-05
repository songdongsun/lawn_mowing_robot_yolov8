"""Run under Ubuntu-22.04 WSL; use the existing standalone AM970 toolchain."""
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
sdk = Path('/root/cpp_project/am970/vknn_demo_yhc/sdk')
toolchain = Path('/sdk/menghai_amicro/output/host')
sysroot = toolchain / 'aarch64-buildroot-linux-gnu/sysroot'
vendor = root / 'third_party/bytetrack'
out = root / 'build'
out.mkdir(exist_ok=True)
common = [str(toolchain / 'bin/aarch64-buildroot-linux-gnu-g++'), '-std=c++17', '-O2', '-Wall', '-Wextra',
          '-I' + str(vendor), '-I' + str(sysroot / 'usr/include/eigen3'), '-I' + str(sdk / 'include'),
          '-I' + str(sysroot / 'usr/include')]
for name, source, libs in [('tracker_check', 'tracker_check.cpp', []),
                           ('detectTrack', 'detect_track.cpp', ['-lvknn', '-lNBGlinker', '-lVIPhal', '-lopencv_videoio'])]:
    args = common + [str(root / source)] + [str(p) for p in sorted(vendor.glob('*.cpp'))]
    args += ['-L' + str(sdk / 'lib'), '-L' + str(sysroot / 'usr/lib'),
             '-Wl,-rpath-link,' + str(sysroot / 'usr/lib')] + libs
    args += ['-lopencv_imgcodecs', '-lopencv_core', '-lopencv_imgproc', '-lpthread', '-ldl', '-o', str(out / name)]
    (out / (name + '_command.json')).write_text(json.dumps(args, indent=2))
    with (out / (name + '_build.log')).open('w') as log:
        subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True)
    print('Built', name, flush=True)
