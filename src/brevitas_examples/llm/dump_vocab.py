# dump_vocab.py
from transformers import AutoTokenizer
from pathlib import Path

model = "meta-llama/Llama-3.2-1B"
tok = AutoTokenizer.from_pretrained(model)

vocab = tok.get_vocab()  # dict: token -> id
items = sorted(vocab.items(), key=lambda x: x[1])  # sort by id

out_path = Path("parameters/embedding_look_up.h")

with out_path.open("w", encoding="utf-8") as f:
    f.write("#ifndef EMBEDDING_LOOK_UP_H_\n")
    f.write("#define EMBEDDING_LOOK_UP_H_\n\n")
    f.write("#include <unordered_map>\n")
    f.write("#include <string>\n\n")
    f.write("static const std::unordered_map<std::string,int> VOCAB = {\n")

    for token, idx in items:
        # escape quotes/backslashes
        safe = token.replace("\\", "\\\\").replace("\"", "\\\"")
        f.write(f"    {{\"{safe}\", {idx}}},\n")

    f.write("};\n\n")
    f.write("#endif // EMBEDDING_LOOK_UP_H_\n")

print(f"[OK] wrote {out_path} with {len(items)} tokens")
