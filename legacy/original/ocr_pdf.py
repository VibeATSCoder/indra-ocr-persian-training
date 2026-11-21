"""Run a trained Indra model on a PDF or image using its original pipeline."""
import argparse
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
source = root/"vendor/indra"
if source.exists():
    sys.path.insert(0,str(source))
import indra_ocr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--model-dir",required=True,help="Folder containing the chosen Indra ONNX model bundle")
    parser.add_argument("--out",required=True)
    parser.add_argument("--mode",choices=["auto","ocr","text"],default="auto")
    parser.add_argument("--cores",type=int,default=4)
    args = parser.parse_args()
    converter = indra_ocr.load(path=args.model_dir,threads=args.cores,mode=args.mode)
    text = "\n\n".join(page.markdown for _,page in converter.convert(args.input))
    Path(args.out).write_text(text,encoding="utf-8")
    print("Saved",args.out)
