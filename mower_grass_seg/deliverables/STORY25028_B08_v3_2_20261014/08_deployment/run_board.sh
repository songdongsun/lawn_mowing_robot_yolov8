#!/bin/sh
set -eu
cd /root/mower_b08_v3_2_20261014
export LD_LIBRARY_PATH=/root/deploy_mower_cam_vknn/lib:/usr/lib/amicro_ndk:/usr/lib
sha256sum /root/deploy_mower_cam_vknn/model/network_binary.nb /root/mower_uint8_trial/network_binary.nb > deployment_before.txt
./mowerConsistencyDump input_list.txt > board_run.log 2>&1
sha256sum network_binary.nb inputs/*.png outputs/*.f32 > board_checksums.txt
sha256sum /root/deploy_mower_cam_vknn/model/network_binary.nb /root/mower_uint8_trial/network_binary.nb > deployment_after.txt
cmp deployment_before.txt deployment_after.txt
tar cf results.tar outputs board_run.log board_checksums.txt deployment_before.txt deployment_after.txt
