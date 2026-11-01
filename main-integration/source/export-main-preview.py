"""Add CPU-only previews of the exact validated main-integration source trees."""

import json
import shutil
from pathlib import Path
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "work/main-integration"
OUT = ROOT / "work/docs-preview-public"
PAGES = {"getting_started/installation/", "getting_started/installation/cpu/"}
NUMBERS = {"erf": 64078, "bf16": 64079, "openmp": 64080}
commits = json.loads((HERE / "prepared-publication-commits.json").read_text())
checks = {x["variant"]: x for x in json.loads((HERE / "candidate-check-results.json").read_text())}
manifest = json.loads((OUT / "manifest.json").read_text())
links = []
for record in commits:
    variant = record["variant"]
    check = checks[variant]
    assert check["index_tree"] == record["tree"]
    assert check["strict_api_docs_exit_code"] == 0 and check["api_generation_enabled"]
    site = HERE / f"site-{variant}"
    for directory in ("assets", "mkdocs/javascript", "mkdocs/stylesheets"):
        shutil.copytree(site / directory, OUT / directory, dirs_exist_ok=True)
    workers = list((site / "assets/javascripts/workers").glob("search.*.min.js"))
    assert len(workers) == 1
    number = NUMBERS[variant]
    name = f"{number}-main"
    for page in sorted(PAGES):
        soup = BeautifulSoup((site / page / "index.html").read_text(), "html.parser")
        for tag in soup.find_all(True):
            for attribute in ("href", "src", "action"):
                value = tag.get(attribute)
                if not value or value.startswith(("#", "data:", "mailto:", "javascript:")):
                    continue
                url = urlsplit(urljoin("https://preview.invalid/" + page, value))
                if url.hostname != "preview.invalid":
                    continue
                relative = url.path.lstrip("/")
                if relative.startswith(("assets/", "mkdocs/")):
                    target = "/vllm/" + relative
                elif relative in PAGES:
                    target = f"/vllm/{name}/" + relative
                else:
                    target = "https://docs.vllm.ai/en/latest/" + relative
                tag[attribute] = target + ("?" + url.query if url.query else "") + ("#" + url.fragment if url.fragment else "")
        config_tag = soup.select_one("script#__config")
        config = json.loads(config_tag.string)
        config["base"] = f"/vllm/{name}/"
        config["search"] = "/vllm/" + str(workers[0].relative_to(site))
        config["features"] = [x for x in config["features"] if not x.startswith("navigation.instant")]
        config_tag.string = json.dumps(config)
        for canonical in soup.select('link[rel="canonical"]'):
            canonical.decompose()
        banner = soup.new_tag("aside")
        banner["style"] = "padding:16px;background:#fff1c2;color:#222;position:relative;z-index:5;font-size:16px"
        banner.string = (
            f"Independent source snapshot for draft PR #{number}: {record['head']}. "
            "Complete strict MkDocs/API build passed. Upstream base: e0d00e4b8eb0c89ec30921d0695cfdd409e9c211. "
            "This is not upstream CI approval. Other documentation links open official latest documentation. "
        )
        link = soup.new_tag("a", href=f"https://github.com/luke-opus/vllm/commit/{record['head']}")
        link.string = "Exact source commit"
        banner.append(link)
        soup.body.insert(0, banner)
        destination = OUT / name / page / "index.html"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(str(soup))
    index = json.loads((site / "search/search_index.json").read_text())
    index["docs"] = [x for x in index["docs"] if x["location"].split("#")[0] in PAGES]
    search = OUT / name / "search/search_index.json"
    search.parent.mkdir(parents=True, exist_ok=True)
    search.write_text(json.dumps(index))
    manifest[name] = {"head": record["head"], "tree": record["tree"], "pages": sorted(PAGES),
                      "strict_api_build": check, "scope": "Pinned source preview; not upstream CI"}
    links.append(f'<li><a href="{name}/getting_started/installation/cpu/">PR #{number}: {variant}</a> <code>{record["head"][:12]}</code></li>')
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
(OUT / "index.html").write_text(
    '<!doctype html><html lang="en"><meta charset="utf-8"><title>vLLM RISC-V draft documentation</title>'
    '<body style="max-width:900px;margin:3em auto;padding:1em;font:18px/1.6 system-ui">'
    '<h1>vLLM RISC-V draft documentation</h1><p>Independent CPU documentation previews from complete strict/API builds on pinned upstream main. '
    'Native hardware, upstream CI and human review remain separate requirements.</p><ul>' + ''.join(links) + '</ul>'
    '<p>Only CPU installation and its installation index are hosted here. Other links open official latest documentation. '
    'Earlier snapshot URLs are retained for historical evidence. Apache-2.0; source attribution is retained.</p></body></html>'
)
print(json.dumps({"exported": list(NUMBERS), "manifest": str(OUT / "manifest.json")}, indent=2))
