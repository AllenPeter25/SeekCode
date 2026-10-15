"""Export the three reference documents using locally installed browser assets.

This development utility is separate from the audio application. It uses
Mistune, Streamlit's bundled KaTeX/Mermaid, and headless Chrome or Edge.
"""

import base64
import html
import json
import re
import subprocess
import tempfile
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import mistune
import requests
import streamlit
import websocket


PROJECT_DIR = Path(__file__).resolve().parents[1]
DOCUMENTS = [
    ("TECHNICAL_REPORT.md", "report", "Part I — Technical report"),
    ("LEARNING_GUIDE.md", "guide", "Part II — Learning guide"),
    ("METHOD_REFERENCE.md", "methods", "Part III — Method reference"),
]
OUTPUT = PROJECT_DIR / "docs" / "seekaudio_complete_reference.pdf"
ASSETS = Path(streamlit.__file__).parent / "static" / "static"


def slug(text):
    text = html.unescape(re.sub(r"<[^>]+>", "", text)).lower()
    return re.sub(r"[^\w\- ]", "", text).replace(" ", "-")


class ReferenceRenderer(mistune.HTMLRenderer):
    def __init__(self, prefix):
        super().__init__(escape=False)
        self.prefix = prefix

    def heading(self, text, level, **attrs):
        anchor = f"{self.prefix}-{slug(text)}"
        return f'<h{level} id="{anchor}">{text}</h{level}>\n'

    def link(self, text, url, title=None):
        if url.startswith("#"):
            url = f"#{self.prefix}-{url[1:]}"
        elif url.split("#")[0] in {item[0] for item in DOCUMENTS}:
            filename, _, anchor = url.partition("#")
            target = next(prefix for name, prefix, _ in DOCUMENTS if name == filename)
            url = f"#{target}-{anchor}" if anchor else f"#part-{target}"
        elif not urlparse(url).scheme:
            url = (PROJECT_DIR / "docs" / url).resolve().as_uri()
        return super().link(text, url, title)

    def block_code(self, code, info=None):
        if info and info.strip() == "mermaid":
            return '<div class="diagram mermaid">' + html.escape(code) + "</div>\n"
        return super().block_code(code, info)

    def block_math(self, text):
        return '<div class="equation" data-math="' + html.escape(text, quote=True) + '"></div>\n'

    def inline_math(self, text):
        return '<span data-math="' + html.escape(text, quote=True) + '"></span>'


def build_html():
    parts = []
    for filename, prefix, title in DOCUMENTS:
        renderer = ReferenceRenderer(prefix)
        markdown = mistune.create_markdown(renderer=renderer, plugins=["table", "math", "strikethrough"])
        renderer.register("block_math", lambda _, text: renderer.block_math(text))
        renderer.register("inline_math", lambda _, text: renderer.inline_math(text))
        source = (PROJECT_DIR / "docs" / filename).read_text(encoding="utf-8")
        parts.append(f'<section id="part-{prefix}" class="part"><div class="part-label">{title}</div>{markdown(source)}</section>')
    katex = next((ASSETS / "js").glob("katex.*.js")).name
    mermaid = next((ASSETS / "js").glob("mermaid.core.*.js")).name
    stylesheet = next((ASSETS / "css").glob("index.*.css")).name
    template = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>seekaudio — Complete technical reference</title>
<link rel="stylesheet" href="/assets/css/__CSS__">
<style>
@page { size: A4; }
* { box-sizing: border-box; }
html, body { background: white !important; color: #172b3b !important; }
body { margin: 0; font: 11pt/1.5 "Segoe UI", Arial, sans-serif; }
.cover { height: 245mm; display: flex; flex-direction: column; justify-content: center; break-after: page; }
.cover .eyebrow, .part-label { color: #18788d; font-size: 11pt; letter-spacing: 1px; text-transform: uppercase; }
.cover h1 { font-size: 44pt; margin: 8mm 0 3mm; color: #133b54; }
.cover h2 { font-size: 23pt; border: 0; margin: 0 0 10mm; }
.cover p { max-width: 145mm; }
.cover nav { margin-top: 12mm; padding: 7mm; border-left: 4px solid #18788d; background: #f1f7f9; }
.cover nav a { display: block; margin: 3mm 0; }
.part + .part { break-before: page; }
.part-label { margin-bottom: 3mm; }
h1 { font-size: 25pt; line-height: 1.2; margin: 0 0 8mm; }
h2 { font-size: 17pt; line-height: 1.25; margin: 9mm 0 4mm; padding-top: 3mm; border-top: 1px solid #cfdee5; }
h3 { font-size: 12.5pt; margin: 6mm 0 2mm; color: #135873; }
h1, h2, h3, h4 { break-after: avoid; }
p { margin: 0 0 3mm; orphans: 3; widows: 3; }
li { margin-bottom: 1.5mm; }
ul, ol { padding-left: 6mm; margin: 2mm 0 4mm; }
a { color: #12627f; text-decoration: none; overflow-wrap: anywhere; }
blockquote { border-left: 3px solid #6da8b7; background: #f3f8fa; padding: 4mm 5mm; margin: 4mm 0; }
code { font: 9pt Consolas, monospace; background: #eef3f6; padding: 0.2mm 0.6mm; overflow-wrap: anywhere; }
pre { background: #f1f5f7; border: 1px solid #dbe5ea; border-radius: 2mm; padding: 3mm; white-space: pre-wrap; overflow-wrap: anywhere; font: 8.8pt/1.4 Consolas, monospace; }
pre code { background: transparent; padding: 0; font: inherit; }
table { width: 100%; border-collapse: collapse; table-layout: auto; margin: 4mm 0 5mm; font-size: 9pt; line-height: 1.35; }
thead { display: table-header-group; }
th { background: #e9f2f6; color: #173f53; text-align: left; }
th, td { border: 1px solid #ccdce3; padding: 2mm; vertical-align: top; overflow-wrap: anywhere; }
tr { break-inside: avoid; }
td code, th code { font-size: 8.3pt; }
.equation { margin: 4mm 0; break-inside: avoid; font-size: 11pt; }
.katex { color: #172b3b; }
.katex-display { margin: 0; overflow: visible; }
.diagram { margin: 5mm auto; text-align: center; break-inside: avoid; }
.diagram svg { max-width: 100% !important; max-height: 225mm; width: auto; height: auto; }
</style></head><body><div id="root" style="display:none"></div>
<section class="cover"><div class="eyebrow">AI · Analytics · Music Information Retrieval</div>
<h1>seekaudio</h1><h2>Complete technical reference<br>and personal learning guide</h2>
<p>A detailed explanation of the implemented audio-analysis and content-based music-retrieval system: theory, formulas, code flow, evaluation evidence, and a from-scratch study path.</p>
<p>Implementation snapshot: 12 October 2026<br>Prepared from the project's annotated Python code and saved version 2 results.</p>
<nav><a href="#part-report">Part I — Technical report: 24 chapters</a><a href="#part-guide">Part II — Learning guide: 12 progressive stages</a><a href="#part-methods">Part III — Method reference: 96 functions/classes</a></nav>
<p style="margin-top:12mm;font-size:9pt;color:#607482">Catalog retrieval scores are not held-out genre-classification accuracy. Future research ideas are distinguished from existing functionality.</p>
</section>
__PARTS__
<script type="module">
window.renderStatus = {ready:false, errors:[]};
try {
 const {t:katex} = await import('/assets/js/__KATEX__');
 const {default:mermaid} = await import('/assets/js/__MERMAID__');
 for (const element of document.querySelectorAll('[data-math]')) {
   katex.render(element.dataset.math, element, {displayMode:element.classList.contains('equation'), throwOnError:true, strict:false});
 }
 mermaid.initialize({startOnLoad:false, theme:'base', securityLevel:'strict', themeVariables:{fontFamily:'Segoe UI, Arial', fontSize:'17px', primaryColor:'#e8f3f6', primaryTextColor:'#172b3b', primaryBorderColor:'#4e8b9e', lineColor:'#4e8b9e'}, flowchart:{htmlLabels:false, useMaxWidth:false, nodeSpacing:18, rankSpacing:28}});
 await mermaid.run({querySelector:'.mermaid', suppressErrors:false});
 await document.fonts.ready;
 window.renderStatus.mathCount = document.querySelectorAll('[data-math]').length;
 window.renderStatus.diagramCount = document.querySelectorAll('.mermaid svg').length;
 window.renderStatus.ready = true;
} catch (error) { window.renderStatus.errors.push(error?.stack || error?.message || JSON.stringify(error)); }
</script></body></html>'''
    return (template.replace("__CSS__", stylesheet).replace("__KATEX__", katex)
            .replace("__MERMAID__", mermaid).replace("__PARTS__", "\n".join(parts)))


class LocalAssetsHandler(SimpleHTTPRequestHandler):
    document = b""

    def do_GET(self):
        if urlparse(self.path).path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(self.document)))
            self.end_headers()
            self.wfile.write(self.document)
            return
        if not urlparse(self.path).path.startswith("/assets/"):
            self.send_error(404)
            return
        super().do_GET()

    def translate_path(self, path):
        relative = unquote(urlparse(path).path).removeprefix("/assets/")
        target = (ASSETS / relative).resolve()
        if not target.is_relative_to(ASSETS.resolve()):
            return str(ASSETS / "not-found")
        return str(target)

    def log_message(self, format, *args):
        pass


class BrowserProtocol:
    def __init__(self, address):
        self.socket = websocket.create_connection(address, timeout=60, suppress_origin=True)
        self.sequence = 0

    def call(self, method, **params):
        self.sequence += 1
        self.socket.send(json.dumps({"id": self.sequence, "method": method, "params": params}))
        while True:
            response = json.loads(self.socket.recv())
            if response.get("id") == self.sequence:
                if "error" in response:
                    raise RuntimeError(response["error"])
                return response.get("result", {})


def main():
    browsers = [Path("C:/Program Files/Google/Chrome/Application/chrome.exe"),
                Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")]
    browser = next((path for path in browsers if path.exists()), None)
    if browser is None:
        raise RuntimeError("Install Chrome or Edge to export this PDF")
    LocalAssetsHandler.document = build_html().encode("utf-8")
    server = ThreadingHTTPServer(("127.0.0.1", 0), LocalAssetsHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory(prefix="seekaudio-pdf-", ignore_cleanup_errors=True) as profile:
        process = subprocess.Popen([str(browser), "--headless=new", "--disable-gpu",
            "--no-first-run", "--no-default-browser-check", "--remote-debugging-port=0",
            f"--user-data-dir={profile}", "about:blank"], stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
        protocol = None
        try:
            port_file = Path(profile) / "DevToolsActivePort"
            deadline = time.monotonic() + 30
            while not port_file.exists():
                if time.monotonic() > deadline or process.poll() is not None:
                    raise RuntimeError("Headless browser did not start")
                time.sleep(0.2)
            port = port_file.read_text().splitlines()[0]
            pages = requests.get(f"http://127.0.0.1:{port}/json", timeout=10).json()
            protocol = BrowserProtocol(next(page["webSocketDebuggerUrl"] for page in pages if page["type"] == "page"))
            protocol.call("Page.enable")
            protocol.call("Page.navigate", url=f"http://127.0.0.1:{server.server_port}/")
            deadline = time.monotonic() + 60
            while True:
                result = protocol.call("Runtime.evaluate", expression="window.renderStatus", returnByValue=True)
                status = result.get("result", {}).get("value", {})
                if status.get("errors"):
                    raise RuntimeError(status["errors"])
                if status.get("ready"):
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError("Timed out rendering math/diagrams")
                time.sleep(0.25)
            footer = '<div style="font:9px Arial;width:100%;padding:0 18mm;color:#607482;display:flex;justify-content:space-between"><span>seekaudio | Technical reference and learning guide</span><span><span class="pageNumber"></span> / <span class="totalPages"></span></span></div>'
            result = protocol.call("Page.printToPDF", printBackground=True,
                paperWidth=8.2677165, paperHeight=11.6929134, marginTop=0.62,
                marginBottom=0.65, marginLeft=0.65, marginRight=0.65,
                displayHeaderFooter=True, headerTemplate="<div></div>", footerTemplate=footer,
                generateDocumentOutline=True, generateTaggedPDF=True)
            OUTPUT.write_bytes(base64.b64decode(result["data"]))
            print(f"Saved {OUTPUT}")
            print(f"Rendered {status['mathCount']} math expressions and {status['diagramCount']} flow diagrams")
        finally:
            if protocol:
                protocol.socket.close()
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            server.shutdown()


if __name__ == "__main__":
    main()
