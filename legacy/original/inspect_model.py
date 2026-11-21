import concurrent.futures
import json
from pathlib import Path
import requests
import onnx
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parent
REV = "bd02167fdf2005a19962cc15188f8496a2c87d3b"

def fetch(name):
    p = ROOT / "base-model" / name
    p.parent.mkdir(exist_ok=True)
    if not p.exists():
        response = requests.get(f"https://huggingface.co/datalab-to/indra-ocr/resolve/{REV}/{name}", timeout=120)
        response.raise_for_status()
        p.write_bytes(response.content)
    return p

if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(3) as pool:
        list(pool.map(fetch, ["rec_rtl.onnx", "meta.json", "LICENSE"]))
    model = onnx.load(ROOT / "base-model/rec_rtl.onnx")
    print("inputs", [(i.name, str(i.type)) for i in model.graph.input])
    print("outputs", [(i.name, str(i.type)) for i in model.graph.output])
    print("weights", [(i.name, list(i.dims)) for i in model.graph.initializer])
    for i, node in enumerate(model.graph.node):
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        if node.op_type == "Constant":
            attrs = {k: numpy_helper.to_array(v).tolist() if isinstance(v, onnx.TensorProto) else v for k,v in attrs.items()}
        print(i, node.op_type, list(node.input), "->", list(node.output), attrs)
