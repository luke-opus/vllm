"""Check copied runtime identity and real package APIs; does not load a model."""

import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

root = Path("/home/vllmtest/vllm-validation")
out = root / "main-integration-runtime"
assert Path(sys.prefix) == root / "main-venv"
assert sys.base_prefix == "/usr"
old = root / ".venv"
new = root / "main-venv"
regular = 0
shared = []
for path in old.rglob("*"):
    if path.is_file() and not path.is_symlink():
        target = new / path.relative_to(old)
        if target.is_file() and not target.is_symlink():
            regular += 1
            a, b = path.stat(), target.stat()
            if (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino):
                shared.append(str(path.relative_to(old)))
assert not shared, shared
(out / "inode-isolation.json").write_text(json.dumps({
    "common_regular_files_checked": regular,
    "shared_regular_inodes": shared,
}, indent=2) + "\n")

import torch
import transformers
import pytest
import pytest_asyncio
import xgrammar
import llguidance
import openai_harmony
import openai_harmony.openai_harmony
import tokenizers
import huggingface_hub
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

assert torch.__version__ == "2.14.0+cpu"
assert transformers.__version__ == "5.22.0"
assert str(new) in torch.__file__ and str(new) in transformers.__file__
assert torch.ones(2, dtype=torch.bfloat16).sum().item() == 2
xgrammar.Grammar.from_regex("a+")
llguidance.LLMatcher.grammar_from_regex("a+")
names = ["torch", "transformers", "pytest", "pytest-asyncio", "xgrammar",
         "llguidance", "oss-harmony", "tokenizers", "huggingface-hub",
         "mistral-common", "safetensors"]
report = {
    "sys_prefix": sys.prefix,
    "sys_base_prefix": sys.base_prefix,
    "python_executable": sys.executable,
    "old_new_common_regular_files_checked": regular,
    "old_new_shared_regular_inodes": shared,
    "package_versions": {n: importlib.metadata.version(n) for n in names},
    "torch_file": torch.__file__,
    "transformers_file": transformers.__file__,
    "checks": ["real Torch BF16 tensor operation", "AutoConfig/AutoTokenizer/AutoModelForCausalLM imports",
               "real XGrammar regex API", "real LLGuidance regex API", "Harmony native module import",
               "pytest and pytest_asyncio imports"],
    "scope": "Selected CPU-test/runtime dependencies only; no vLLM DSO, model, or full test suite execution",
}
(out / "runtime-check.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
