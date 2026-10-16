"""Copy candidate obstacle images and original labels into class/split review folders."""

import argparse
import ast
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path


CHINESE = {
    "garden hose": "软管", "garden_hose": "软管", "branch": "树枝",
    "Rock": "岩石", "rock": "石块", "axe": "斧头", "brush-cutter": "割灌机",
    "gardening fork": "园艺叉", "grass shear": "草剪", "hedge-shear": "绿篱剪",
    "hoe": "锄头", "lawn-rake": "草坪耙", "machete": "砍刀", "picklaxe": "镐",
    "pruning-saw": "修枝锯", "rake": "耙子", "shovel": "铲子", "sickel": "镰刀",
    "spade": "铁锹", "sprayers": "喷雾器", "trowel": "园艺手铲",
    "watering-cane": "浇水壶", "wheel-barrow": "手推车",
    "flowerpot": "花盆", "watering_can": "浇水壶", "bucket": "水桶",
}
SPLITS = {"train": "训练集_train", "valid": "验证集_valid", "test": "测试集_test"}


def copy_verified(source, destination):
    """Make an independent copy and verify contents, without changing the source."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(destination.read_bytes()).digest():
        raise ValueError(f"Copy verification failed: {destination}")


def organize(root):
    output = root / "按类别人工审核"
    if output.exists():
        raise ValueError(f"Refusing to overwrite existing review directory: {output}")
    rows = []
    for dataset in ("Obstructions.v1i.yolov8", "Roboflow_lawn-10.1_v2"):
        source = root / dataset
        names = ast.literal_eval(next(
            line.split(":", 1)[1] for line in (source / "data.yaml").read_text().splitlines()
            if line.startswith("names:")
        ))
        for split, split_name in SPLITS.items():
            grouped = defaultdict(list)
            images = {p.stem: p for p in (source / split / "images").iterdir()} if (source / split / "images").exists() else {}
            for label in sorted((source / split / "labels").glob("*.txt")):
                target_counts = defaultdict(int)
                for line in label.read_text().splitlines():
                    fields = line.split()
                    if fields:
                        name = names[int(fields[0])]
                        if name in CHINESE:
                            target_counts[name] += 1
                for name, count in target_counts.items():
                    grouped[name].append((images[label.stem], label, count))
            for name, items in grouped.items():
                destination = output / f"{CHINESE[name]}_{name}" / dataset / split_name
                for image, label, _ in items:
                    copy_verified(image, destination / "images" / image.name)
                    copy_verified(label, destination / "原始标签" / label.name)
                copy_verified(source / "data.yaml", destination.parent / "原始类别配置_仅供查阅.yaml")
                rows.append({"类别": name, "中文名": CHINESE[name], "来源": dataset, "划分": split_name,
                             "图片数": len(items), "目标标注数": sum(i[2] for i in items),
                             "目录": str(destination.relative_to(output))})
    source = next(root.glob("LVIS_*"))
    data = json.loads((source / "selected_original_annotations.json").read_text(encoding="utf-8"))
    categories = {c["id"]: c for c in data["categories"]}
    existing = {int(p.stem): p for p in (source / "images").glob("*.jpg")}
    grouped = defaultdict(list)
    for annotation in data["annotations"]:
        name = categories[annotation["category_id"]]["name"]
        if name in {"garden_hose", "flowerpot", "watering_can", "shovel", "bucket"} and annotation["image_id"] in existing:
            grouped[name].append(annotation)
    for name, annotations in grouped.items():
        destination = output / f"{CHINESE[name]}_{name}" / source.name / "验证集_val"
        ids = {a["image_id"] for a in annotations}
        for image_id in sorted(ids):
            copy_verified(existing[image_id], destination / "images" / existing[image_id].name)
        subset = {
            "info": data.get("info", {}), "licenses": data.get("licenses", []),
            "categories": [c for c in data["categories"] if c["name"] == name],
            "images": [i for i in data["images"] if i["id"] in ids], "annotations": annotations,
        }
        (destination / "原始类别标注子集.json").write_text(json.dumps(subset, ensure_ascii=False), encoding="utf-8")
        rows.append({"类别": name, "中文名": CHINESE[name], "来源": source.name, "划分": "验证集_val",
                     "图片数": len(ids), "目标标注数": len(annotations), "目录": str(destination.relative_to(output))})
    with (output / "分类数量统计.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "审核说明.txt").write_text(
        "本目录仅用于人工审核，不是可直接训练的数据集。原数据未改动；图片是独立副本。\n"
        "按中文名_原始标签名 / 来源数据集 / 原始划分 / images组织。\n"
        "Rock与rock分别放在岩石_Rock、石块_rock，避免Windows大小写目录冲突。\n"
        "训练/验证/测试沿用来源划分；LVIS子集来自val，未人为重分。\n"
        "YOLO原始标签整份复制，可能含其他类；ID和名称不修改，类别配置仅供查阅。\n"
        "LVIS JSON只筛选当前类别及对应图片，原始标注内容、ID、bbox和分割信息不变。\n"
        "一张图含多种目标时会出现在多个类别目录，统计不可直接相加作为唯一图片数。\n"
        "Obstructions混合检测框与多边形；多边形不能直接作为五列检测框使用。\n"
        "lawn-10.1的branch存在框落在路灯杆上的疑似错标，必须逐张审核；Tree未当作branch。\n"
        "包含增强版本、商品图和非草坪场景，尚未清洗或判断全部样本合格。\n"
        "水桶、割灌机、手推车等是待确认候选，不代表已决定最终类别范围。\n"
        "未找到线缆类别；未纳入人员动物、树木、植物、靴子和拖拉机等其他类别。\n"
        "合并训练前需复核漏标、统一标签格式与ID，并防止同源图跨训练验证集泄漏。\n",
        encoding="utf-8-sig",
    )
    print(json.dumps({"folders": len(rows), "image_copies": sum(r["图片数"] for r in rows),
                      "categories": len({r["类别"] for r in rows})}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    organize(parser.parse_args().root.resolve())
