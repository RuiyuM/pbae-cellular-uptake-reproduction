"""Download the paper's public sources; record URLs and SHA256 checksums."""
from pathlib import Path
import hashlib
import json
import requests

ROOT = Path(__file__).resolve().parents[1]
session = requests.Session()
manifest = []

def download(url, destination, pdf=False):
    destination.parent.mkdir(parents=True, exist_ok=True)
    r = session.get(url, timeout=120)
    r.raise_for_status()
    if pdf and not r.content.startswith(b'%PDF'):
        raise ValueError('Response is not a PDF')
    destination.write_bytes(r.content)
    manifest.append({'url': url, 'file': str(destination.relative_to(ROOT)),
                     'bytes': len(r.content), 'sha256': hashlib.sha256(r.content).hexdigest()})
    print(destination.name, len(r.content), flush=True)
    return r.text if not pdf else None

papers = {
    'paper.pdf': [
        'https://pubs.rsc.org/en/content/articlepdf/2023/bm/d3bm00741c',
        'https://dau.url.edu/bitstream/handle/20.500.14342/4643/Biomat_scie_2023-11_a-machine.pdf?sequence=1',
    ],
    'supplement.pdf': ['https://www.rsc.org/suppdata/d3/bm/d3bm00741c/d3bm00741c1.pdf'],
}
for name, urls in papers.items():
    for url in urls:
        try:
            download(url, ROOT/'sources'/name, pdf=True)
            break
        except Exception as e:
            print(name, type(e).__name__, str(e)[:160], flush=True)

for acc in ['GSM1406256', 'GSM1529765', 'GSM564167', 'GSM613188']:
    text = download('https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc='+acc+'&targ=self&form=text&view=full',
                    ROOT/'data/raw'/(acc+'.soft'))
    for line in text.splitlines():
        if line.startswith('!Sample_supplementary_file'):
            url = line.split(' = ', 1)[1].replace('ftp://', 'https://')
            if url.startswith('https://ftp.ncbi.nlm.nih.gov/'):
                download(url, ROOT/'data/raw'/url.rsplit('/', 1)[-1])
    (ROOT/'sources/download_manifest.json').write_text(json.dumps(manifest, indent=2))

(ROOT/'sources/download_manifest.json').write_text(json.dumps(manifest, indent=2))
