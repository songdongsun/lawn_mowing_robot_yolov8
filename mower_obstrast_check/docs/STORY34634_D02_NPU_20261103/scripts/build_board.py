"""Cross-compile against the existing standalone AM970 SDK, without Docker/company host."""
import json
from pathlib import Path
import subprocess

OUT=Path(__file__).resolve().parents[1]
sdk=Path('/root/cpp_project/am970/vknn_demo_yhc/sdk')
toolchain=Path('/sdk/menghai_amicro/output/host')
sysroot=toolchain/'aarch64-buildroot-linux-gnu/sysroot'
args=[str(toolchain/'bin/aarch64-buildroot-linux-gnu-g++'),'-std=c++17','-O2','-Wall','-Wextra',
      '-I'+str(sdk/'include'),'-I'+str(sysroot/'usr/include'),str(OUT/'scripts/detect_static.cpp'),
      '-L'+str(sdk/'lib'),'-L'+str(sysroot/'usr/lib'),'-Wl,-rpath-link,'+str(sysroot/'usr/lib'),
      '-lvknn','-lNBGlinker','-lVIPhal','-lopencv_imgcodecs','-lopencv_core','-lopencv_imgproc','-lpthread','-ldl',
      '-o',str(OUT/'board/detectStatic')]
(OUT/'board/build_command.json').write_text(json.dumps(args,indent=2))
with (OUT/'board/build.log').open('w') as log:
    subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True)
print('Cross-compile complete')
