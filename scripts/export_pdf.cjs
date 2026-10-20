/** Print the self-contained report to A4, retaining Chinese text and hyperlinks.
 * Optional export dependency: npm install --no-save playwright
 * Then: npx playwright install chromium; node scripts/export_pdf.cjs
 * CHROME_PATH may point to an existing Chrome executable instead.
 */
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
let chromium;
try { ({ chromium } = require('playwright')); }
catch { ({ chromium } = require('playwright-core')); }
const root = path.resolve(__dirname, '..');
const repo = 'https://github.com/RuiyuM/pbae-cellular-uptake-reproduction';
const output = path.join(root, 'output/pdf/pbae_table2_reproduction_report.pdf');

async function main() {
  fs.mkdirSync(path.dirname(output), { recursive: true });
  const options = { headless: true };
  if (process.env.CHROME_PATH) options.executablePath = process.env.CHROME_PATH;
  const browser = await chromium.launch(options);
  try {
    const page = await browser.newPage({ viewport: { width: 680, height: 1000 } });
    await page.goto(pathToFileURL(path.join(root, 'report.html')).href);
    await page.evaluate(async () => {
      await document.fonts.ready;
      await Promise.all([...document.images].map(img => img.decode()));
    });
    await page.evaluate(({ repo }) => {
      // Each original main section starts a new printed page.
      const main = document.querySelector('main');
      const children = [...main.children];
      let section = document.createElement('section');
      section.className = 'print-section overview';
      main.append(section);
      let count = 0;
      for (const child of children) {
        if (child.tagName === 'H2') {
          count += 1;
          if (count > 1) {
            section = document.createElement('section');
            section.className = 'print-section';
            main.append(section);
          }
        }
        section.append(child);
      }
      for (const anchor of document.querySelectorAll('a[href]')) {
        const href = anchor.getAttribute('href');
        if (!/^(https?:|mailto:|#)/.test(href)) anchor.href = `${repo}/blob/main/${href}`;
      }
      const files = [...document.querySelectorAll('.print-section')].at(-1);
      const guide = document.createElement('div');
      guide.innerHTML = `<h3>获取代码与重新运行</h3>
        <p>打开以下仓库，点击 Code → Download ZIP 即可下载完整代码包。仓库同时保存本报告 PDF、处理后的输入、逐次结果与校验记录。</p>
        <p class="repository"><a href="${repo}">${repo}</a></p>
        <p>运行环境与顺序见仓库首页 README。先下载固定版本的作者仓库及 GEO 原始数据，再依次重建特征、训练、校验和生成报告。原始芯片及作者论文通过来源下载，不随代码包重复打包。</p>
        <p class="small">数据来源、文件校验值和具体假设见 sources/download_manifest.json、results/input_checksums.json 与 results/protocol.json。</p>`;
      files.insertBefore(guide, files.querySelector('footer'));
      document.title = 'pBAE 细胞摄取预测 - 论文复现与 Table 2 对比';
    }, { repo });
    await page.addStyleTag({ content: `
      @page { size: A4; margin: 13mm 14mm 17mm; }
      html, body { background: white; }
      body { font-family: "PingFang SC", "Noto Sans CJK SC", "Microsoft YaHei", sans-serif;
             font-size: 10pt; line-height: 1.55; color: #182a3a;
             -webkit-print-color-adjust: exact; print-color-adjust: exact; }
      main { max-width: none; margin: 0; padding: 0; box-shadow: none; border-radius: 0; }
      .print-section { break-before: page; }
      .print-section:first-child { break-before: auto; }
      h1 { font-size: 23pt; line-height: 1.25; margin: 6px 0 9px; }
      h2 { font-size: 15pt; margin: 0 0 10px; padding: 0; border: 0; break-after: avoid; }
      .overview h2 { margin-top: 12px; }
      h3 { font-size: 11.5pt; margin: 17px 0 5px; break-after: avoid; }
      .eyebrow { font-size: 8pt; letter-spacing: 1.5px; }
      p { margin: 8px 0; orphans: 3; widows: 3; }
      .lead { font-size: 10.5pt; }
      .repository { background: #edf6f7; border-left: 3px solid #087f8c; padding: 7px 10px;
                    font-size: 9pt; line-height: 1.5; overflow-wrap: anywhere; }
      .repository br { display: none; }
      .cards { gap: 9px; margin: 12px 0; grid-template-columns: repeat(4,1fr); }
      .card { padding: 8px 10px; font-size: 8.5pt; border-radius: 6px; }
      .card b { font-size: 17pt; line-height: 1.4; }
      table { font-size: 8.2pt; line-height: 1.45; margin: 10px 0; white-space: normal; }
      th, td { padding: 7px 6px; }
      table, tr, img, pre, .note { break-inside: avoid; }
      thead { display: table-header-group; }
      .scroll { overflow: visible; }
      img { display: block; width: auto; max-width: 100%; height: auto; max-height: 88mm;
            margin: 10px auto; object-fit: contain; }
      img[alt="recovered_gene_expression.png"] { max-height: 77mm; }
      img[alt="nn_learning_curves.png"] { max-height: 112mm; margin-top: 20px; }
      .note { padding: 9px 12px; font-size: 9pt; line-height: 1.5; }
      pre, code { font-size: 8.3pt; }
      pre { white-space: pre-wrap; overflow-wrap: anywhere; padding: 10px 12px; margin: 10px 0; }
      ul { padding-left: 20px; margin: 8px 0; }
      li { margin: 5px 0; }
      .small, footer { font-size: 8.5pt; line-height: 1.55; }
      footer { margin-top: 22px; padding-top: 12px; overflow-wrap: anywhere; }
      a { color: #087f8c; text-decoration: underline; }
    ` });
    await page.emulateMedia({ media: 'print' });
    const layout = await page.locator('.print-section').evaluateAll(nodes => nodes.map(n => ({
      heading: n.querySelector('h1,h2')?.innerText,
      height: Math.round(n.getBoundingClientRect().height)
    })));
    console.log(JSON.stringify(layout, null, 2));
    await page.pdf({ path: output, format: 'A4', printBackground: true,
      preferCSSPageSize: true, displayHeaderFooter: true, headerTemplate: '<span></span>',
      footerTemplate: `<div style="width:100%;padding:0 14mm;display:flex;justify-content:space-between;font:8px Arial;color:#526779"><span>pBAE reproduction · RuiyuM · 2026-10-20</span><span><span class="pageNumber"></span> / <span class="totalPages"></span></span></div>`
    });
    console.log(output);
  } finally { await browser.close(); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
