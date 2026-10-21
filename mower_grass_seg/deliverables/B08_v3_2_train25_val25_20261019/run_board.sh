#!/bin/bash
set -euo pipefail
cd /root/sds/static-b08-train25-val25-20261019
mkdir -p outputs
: > board_outputs.sha256
while IFS= read -r sid; do
    printf '%s\n' "$sid" > single_list.txt
    /root/sds/static-b08-v3-2-20261019/mowerConsistencyDump single_list.txt
    sha256sum "outputs/${sid}_0.f32" "outputs/${sid}_1.f32" >> board_outputs.sha256
    gzip "outputs/${sid}_0.f32" "outputs/${sid}_1.f32"
done < input_list.txt
sha256sum network_binary.nb inputs/*.png > board_inputs.sha256
