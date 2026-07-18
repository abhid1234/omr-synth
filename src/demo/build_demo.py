"""Build the self-contained synthetic-pairs browser demo."""
from __future__ import annotations

import argparse
import base64
import html
import tempfile
from pathlib import Path

from PIL import Image

from src.synth.augment import degrade
from src.synth.render import JianpuRenderer, SargamRenderer, VerovioRenderer
from src.synth.scores import generate_score
from src.vocab.dsl import serialize

CURATED_SAMPLES = (
    (3101, 0, 2), (3107, 0, 2), (3119, 1, 3), (3137, 1, 3),
    (3163, 2, 3), (3181, 2, 3), (3203, 2, 4), (3229, 3, 4),
    (3251, 3, 4), (3271, 3, 5),
)


def _data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _prepare_image(path: Path, seed: int, level: int) -> None:
    degrade(path, seed, level)
    with Image.open(path) as source:
        image = source.convert("L")
        image.thumbnail((1200, 420), Image.Resampling.LANCZOS)
        image.save(path, optimize=True)


def _shared_target(record) -> str:
    tokens = serialize(record, notation="western").split()
    tokens.remove("NOTATION_WESTERN")
    return " ".join(tokens)


def build_demo(output: Path = Path("demo/index.html")) -> Path:
    """Render the curated paired gallery to one offline HTML file."""
    renderers = (VerovioRenderer(), JianpuRenderer(), SargamRenderer())
    cards: list[str] = []
    with tempfile.TemporaryDirectory(prefix="omr-demo-") as directory:
        work = Path(directory)
        for index, (seed, level, measures) in enumerate(CURATED_SAMPLES, 1):
            score, record = generate_score(seed, level=level, measures=measures)
            images: dict[str, str] = {}
            for renderer in renderers:
                path = work / f"{index}-{renderer.notation}.png"
                renderer.render(score, record, path)
                _prepare_image(path, seed, level)
                images[renderer.notation] = _data_uri(path)
            target = html.escape(_shared_target(record))
            degradation = " · manuscript-degraded" if level == 3 else ""
            cards.append(f"""
<article class="sample">
  <div class="sample-head"><h2>Sample {index:02d}</h2><div class="tags"><span>curriculum level {level}</span><span>Western</span><span>Jianpu</span><span>Sargam</span><span>seed {seed}</span></div></div>
  <p class="mode">Same symbolic record · three render grammars{degradation}</p>
  <div class="renders">
    <figure><figcaption>Western staff</figcaption><div class="image"><img src="{images['western']}" alt="Western rendering for sample {index}"></div></figure>
    <figure><figcaption>Jianpu numbered notation</figcaption><div class="image"><img src="{images['jianpu']}" alt="Jianpu rendering for sample {index}"></div></figure>
    <figure><figcaption>Sargam solfège notation</figcaption><div class="image"><img src="{images['sargam']}" alt="Sargam rendering for sample {index}"></div></figure>
  </div>
  <div class="target"><div class="target-label">Shared OMRDSL semantic target</div><code>{target}</code></div>
</article>""")

    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>omr-synth · synthetic pairs gallery</title>
<style>
:root{{--paper:#f6f3ec;--card:#fff;--ink:#182020;--muted:#61706d;--line:#d8ded9;--accent:#176b61;--soft:#e9f3f0}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.55 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}main{{width:min(1180px,calc(100% - 28px));margin:auto;padding:52px 0 72px}}header{{max-width:850px;margin-bottom:34px}}.eyebrow{{color:var(--accent);font-size:.78rem;font-weight:800;letter-spacing:.13em;text-transform:uppercase}}h1{{font-size:clamp(2rem,6vw,4rem);line-height:1.03;letter-spacing:-.045em;margin:.25rem 0 1rem}}header p{{font-size:1.08rem;color:#3d4a48;margin:.5rem 0}}.honesty{{border-left:4px solid var(--accent);padding:10px 15px;background:var(--soft);margin-top:20px}}.sample{{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:22px;margin:18px 0;box-shadow:0 5px 20px #24332d0b}}.sample-head{{display:flex;align-items:center;justify-content:space-between;gap:14px;flex-wrap:wrap}}h2{{font-size:1.15rem;margin:0}}.tags{{display:flex;gap:7px;flex-wrap:wrap}}.tags span{{background:#eef1ef;border:1px solid #dce2de;border-radius:999px;padding:3px 9px;font-size:.76rem;font-weight:700;color:#44514f}}.mode{{margin:7px 0 16px;color:var(--muted);font-size:.88rem}}.renders{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}figure{{margin:0;min-width:0}}figcaption,.target-label{{font-size:.78rem;font-weight:800;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin-bottom:7px}}.image{{height:250px;border:1px solid var(--line);border-radius:10px;background:#fff;display:flex;align-items:center;justify-content:center;overflow:auto;padding:8px}}img{{max-width:100%;max-height:100%;object-fit:contain}}.target{{margin-top:16px;background:#15211f;color:#dce8e4;border-radius:10px;padding:13px 15px}}.target .target-label{{color:#8fc4b9}}code{{display:block;font:12px/1.65 ui-monospace,SFMono-Regular,Menlo,monospace;white-space:pre-wrap;overflow-wrap:anywhere}}footer{{color:var(--muted);text-align:center;margin-top:32px;font-size:.86rem}}@media(max-width:900px){{.renders{{grid-template-columns:1fr}}}}@media(max-width:720px){{main{{padding-top:30px}}.sample{{padding:16px}}.image{{height:210px}}}}
</style></head><body><main>
<header><div class="eyebrow">Synthetic-pairs gallery</div><h1>One score. Three notations. One musical target.</h1>
<p>These are synthetic ground-truth examples produced by the real render-forward engine. Every card starts with one symbolic score record, renders it as Western staff notation, Jianpu, and Sargam, then shows the shared semantic OMRDSL target.</p>
<p class="honesty"><strong>This is not a trained-model prediction.</strong> No model runs in this page. Training and evaluation against real manuscripts are the funded next step.</p></header>
{''.join(cards)}
<footer>Generated locally by omr-synth · embedded images and styles · no network required</footer>
</main></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("demo/index.html"))
    args = parser.parse_args()
    path = build_demo(args.output)
    print(f"generated {len(CURATED_SAMPLES)} paired samples in {path}")


if __name__ == "__main__":
    main()
