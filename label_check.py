from pathlib import Path
import xml.etree.ElementTree as ET
from collections import Counter

ann_dir = Path("data/Annotations")
img_dir = Path("data/JPEGImages")

xml_files = sorted(ann_dir.glob("*.xml"))
jpg_files = sorted(img_dir.glob("*.jpg"))

classes = Counter()
no_object_xml = 0
for xf in xml_files:
    root = ET.parse(xf).getroot()
    objs = root.findall("object")
    if not objs:
        no_object_xml += 1
    for o in objs:
        classes[o.findtext("name", default="UNKNOWN")] += 1

print("images:", len(jpg_files), "xml:", len(xml_files), "xml_with_no_object:", no_object_xml)
print("object classes:", classes)