"""Final evaluation of a trained ECP checkpoint with the authors' own metric code, in fp32.

Loads best.pt or last.pt into the ORIGINAL ECPAttentionTransformer (the bf16 copy has identical parameter names),
disables TF32, and calls the authors' compute_perplexity_huggingface (extracted unmodified from
ECPTransformer_main.py) on the test split, and optionally on the validation split. Single GPU.

    python -m ecp_pg19.eval_ecp_pg19 --ckpt output/ecp24_pg19/best.pt \
        --data_dir /work/.../LLM_DATASETS/pg19/tokens --splits test validation
"""
import argparse
import json
import os
import time

import torch

from ecp_pg19.ecp_base import authors_compute_perplexity
from ecp_pg19.ecp_model import build_ecp
from ecp_pg19.pg19_data import EvalSet


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--splits", nargs="+", default=["test"], choices=["test", "validation"])
    p.add_argument("--out_json", default=None, help="default: <ckpt>.eval_fp32.json")
    p.add_argument("--device", default=None)
    a = p.parse_args(argv)

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    c = ck["config"]
    model = build_ecp(c["layers"], c["dim"], c["heads"], c["sequence_length"], c["num_segments"], c["net_dropout"],
                      c["vocab_size"], variant="original")
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    out = {"ckpt": os.path.abspath(a.ckpt), "iteration": ck.get("iteration"), "precision": "fp32", "metric":
           "authors' compute_perplexity_huggingface (windows of seq_len+1, stride seq_len//2)"}
    for split in a.splits:
        ds = EvalSet(os.path.join(a.data_dir, f"{split}.uint16.bin"), c["sequence_length"])
        t0 = time.time()
        ppl, _ = authors_compute_perplexity(model, ds, device, c["sequence_length"], c["net_dropout"])
        out[f"{split}_ppl"] = ppl
        print(f"{split}: ppl {ppl:.4f} ({time.time() - t0:.0f}s)", flush=True)
    path = a.out_json or a.ckpt + ".eval_fp32.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
