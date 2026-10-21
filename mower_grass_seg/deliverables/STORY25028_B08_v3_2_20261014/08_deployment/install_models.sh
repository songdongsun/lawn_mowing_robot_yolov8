#!/bin/sh
set -eu
SOURCE=/root/mower_b08_v3_2_20261014/network_binary.nb
CAM=/root/deploy_mower_cam_vknn/model/network_binary.nb
STATIC=/root/mower_uint8_trial/network_binary.nb
EXPECTED=9f162de626472adaac621eb5ffde0fff4dc3e7daf6089ec1810c4141023bf898
BACKUP=/root/model_backups/before_b08_v3_2_20261014_$$
actual=$(sha256sum "$SOURCE")
test "${actual%% *}" = "$EXPECTED"
test -f "$CAM"
test -f "$STATIC"
mkdir -p /root/model_backups
mkdir "$BACKUP"
cp -p "$CAM" "$BACKUP/camera_network_binary.nb"
cp -p "$STATIC" "$BACKUP/static_network_binary.nb"
cmp "$CAM" "$BACKUP/camera_network_binary.nb"
cmp "$STATIC" "$BACKUP/static_network_binary.nb"
sha256sum "$CAM" "$STATIC" > "$BACKUP/before.sha256"
rollback() {
    code=$?
    if [ "$code" -ne 0 ]; then
        cp -p "$BACKUP/camera_network_binary.nb" "$CAM"
        cp -p "$BACKUP/static_network_binary.nb" "$STATIC"
        echo "Installation failed; restored originals from $BACKUP" >&2
    fi
}
trap rollback EXIT
cp -p "$SOURCE" "$CAM.new.$$"
cp -p "$SOURCE" "$STATIC.new.$$"
cmp "$SOURCE" "$CAM.new.$$"
cmp "$SOURCE" "$STATIC.new.$$"
mv "$CAM.new.$$" "$CAM"
mv "$STATIC.new.$$" "$STATIC"
cmp "$SOURCE" "$CAM"
cmp "$SOURCE" "$STATIC"
sha256sum "$CAM" "$STATIC" > "$BACKUP/after.sha256"
sync
echo "Installed B08_v3_2; backup: $BACKUP"
cat "$BACKUP/after.sha256"
