const fs = require("fs");
const path = require("path");

const inputPath = path.resolve(
  __dirname,
  "..",
  "..",
  "RL-Kernel",
  "docs",
  "blog",
  "2026-07-08-announcing-rl-kernel-linear-logp-for-vime-zh.md",
);
const outputPath = path.resolve(__dirname, "..", "2026-07-08-announcing-rl-kernel-linear-logp-for-vime-zh-wechat.html");

function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function inlineMarkdown(text) {
  const placeholders = [];
  let s = escapeHtml(text);

  s = s.replace(/`([^`]+)`/g, (_, code) => {
    const token = `@@CODE${placeholders.length}@@`;
    placeholders.push(
      `<span style="font-family: Menlo, Consolas, monospace; font-size: 0.95em; color: #16324f;">${code}</span>`,
    );
    return token;
  });

  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, label, href) => {
    return `<a href="${href}" style="color: #2563eb; text-decoration: none;">${label}</a>`;
  });

  placeholders.forEach((value, index) => {
    s = s.replace(`@@CODE${index}@@`, value);
  });
  return s;
}

function stripFrontMatter(source) {
  if (!source.startsWith("---\n")) return { attrs: {}, body: source };
  const end = source.indexOf("\n---\n", 4);
  if (end < 0) return { attrs: {}, body: source };
  const raw = source.slice(4, end).split(/\r?\n/);
  const attrs = {};
  for (const line of raw) {
    const match = line.match(/^([A-Za-z0-9_-]+):\s*(.*)$/);
    if (!match) continue;
    attrs[match[1]] = match[2].replace(/^"|"$/g, "");
  }
  return { attrs, body: source.slice(end + 5).trim() };
}

function imagePath(src) {
  const name = path.basename(src).replace(/\.svg$/i, ".png");
  if (src.includes("architecture") || src.includes("rlk-global-architecture")) {
    return "wechat-assets/rlk-global-architecture.png";
  }
  if (name === "rlk-linear-logp-dataflow.png") return "wechat-assets/rlk-linear-logp-dataflow.png";
  if (name === "rlk-linear-logp-speedup.png") return "wechat-assets/rlk-linear-logp-speedup.png";
  if (name === "rlk-linear-logp-memory.png") return "wechat-assets/rlk-linear-logp-memory.png";
  return src;
}

function renderTable(lines) {
  const rows = lines
    .filter((line) => !/^\|\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$/.test(line))
    .map((line) => line.trim().replace(/^\||\|$/g, "").split("|").map((cell) => cell.trim()));
  if (!rows.length) return "";
  const [head, ...body] = rows;
  const th = head
    .map((cell) => `<th style="border: 1px solid #d9e2ec; padding: 10px 12px; background: #eef4ff; color: #172033; font-weight: 700; text-align: left; white-space: nowrap;">${inlineMarkdown(cell)}</th>`)
    .join("");
  const tr = body
    .map((row) => {
      const cells = row
        .map((cell) => `<td style="border: 1px solid #d9e2ec; padding: 10px 12px; color: #25324a; vertical-align: top; white-space: nowrap;">${inlineMarkdown(cell)}</td>`)
        .join("");
      return `<tr>${cells}</tr>`;
    })
    .join("\n");
  const minWidth = head.length >= 7 ? 1280 : head.length >= 5 ? 980 : 920;
  return `<section style="display: block; width: 100%; overflow-x: auto; -webkit-overflow-scrolling: touch; margin: 18px 0; padding-bottom: 8px;"><table style="border-collapse: collapse; width: ${minWidth}px; min-width: ${minWidth}px; font-size: 13px; line-height: 1.55;"><thead><tr>${th}</tr></thead><tbody>${tr}</tbody></table></section>`;
}

function renderParagraph(lines) {
  const text = lines.join("\n").trim();
  if (!text) return "";
  return `<p style="margin: 14px 0; color: #25324a; font-size: 16px; line-height: 1.85;">${inlineMarkdown(text).replace(/\n/g, "<br>")}</p>`;
}

function renderMarkdown(body) {
  const lines = body.replace(/\r\n/g, "\n").split("\n");
  const out = [];
  let para = [];
  let list = [];
  let table = [];
  let code = [];
  let inCode = false;
  let inFigure = false;
  let figureSrc = "";
  let figureAlt = "";
  let figureCaption = "";

  function flushPara() {
    if (para.length) out.push(renderParagraph(para));
    para = [];
  }
  function flushList() {
    if (!list.length) return;
    out.push(`<ul style="padding-left: 1.2em; margin: 14px 0 18px; color: #25324a; font-size: 16px; line-height: 1.8;">${list.map((item) => `<li style="margin: 6px 0;">${inlineMarkdown(item)}</li>`).join("")}</ul>`);
    list = [];
  }
  function flushTable() {
    if (table.length) out.push(renderTable(table));
    table = [];
  }
  function flushCode() {
    const text = escapeHtml(code.join("\n")).replace(/\n/g, "<br>");
    out.push(`<section style="display: block; overflow-x: auto; -webkit-overflow-scrolling: touch; margin: 16px 0; padding: 13px 15px; background: #f7f9fc; border: 1px solid #d9e2ec; border-radius: 8px;">
  <p style="margin: 0; color: #25324a; font-family: Menlo, Consolas, monospace; font-size: 13px; line-height: 1.7; white-space: nowrap;">${text}</p>
</section>`);
    code = [];
  }
  function flushFigure() {
    if (!figureSrc) return;
    out.push(`<figure style="margin: 22px 0; text-align: center;">
  <img src="${imagePath(figureSrc)}" alt="${escapeHtml(figureAlt || "")}" style="display: block; width: 100%; max-width: 760px; margin: 0 auto; border-radius: 6px;">
  ${figureCaption ? `<figcaption style="margin-top: 8px; color: #667085; font-size: 13px; line-height: 1.6;">${inlineMarkdown(figureCaption)}</figcaption>` : ""}
</figure>`);
    inFigure = false;
    figureSrc = "";
    figureAlt = "";
    figureCaption = "";
  }

  for (const rawLine of lines) {
    const line = rawLine.trimEnd();

    if (line.startsWith("```")) {
      flushPara();
      flushList();
      flushTable();
      if (inCode) {
        flushCode();
        inCode = false;
      } else {
        inCode = true;
      }
      continue;
    }
    if (inCode) {
      code.push(line);
      continue;
    }

    if (line.trim() === '<p align="center">') {
      flushPara();
      flushList();
      flushTable();
      inFigure = true;
      continue;
    }
    if (inFigure) {
      const img = line.match(/<img[^>]*src="([^"]+)"[^>]*alt="([^"]*)"[^>]*>/);
      if (img) {
        figureSrc = img[1];
        figureAlt = img[2];
        continue;
      }
      const cap = line.match(/<em>(.*)<\/em>/);
      if (cap) {
        figureCaption = cap[1];
        continue;
      }
      if (line.trim() === "</p>") {
        flushFigure();
        continue;
      }
    }

    if (!line.trim()) {
      flushPara();
      flushList();
      flushTable();
      continue;
    }

    if (/^\|.*\|$/.test(line.trim())) {
      flushPara();
      flushList();
      table.push(line);
      continue;
    }
    if (table.length) flushTable();

    const h2 = line.match(/^##\s+(.+)$/);
    if (h2) {
      flushPara();
      flushList();
      out.push(`<h2 style="margin: 32px 0 14px; padding-left: 12px; border-left: 4px solid #2563eb; color: #0f172a; font-size: 22px; line-height: 1.45; font-weight: 800;">${inlineMarkdown(h2[1])}</h2>`);
      continue;
    }
    const h3 = line.match(/^###\s+(.+)$/);
    if (h3) {
      flushPara();
      flushList();
      out.push(`<h3 style="margin: 26px 0 12px; color: #172033; font-size: 18px; line-height: 1.5; font-weight: 800;">${inlineMarkdown(h3[1])}</h3>`);
      continue;
    }
    const bullet = line.match(/^-\s+(.+)$/);
    if (bullet) {
      flushPara();
      list.push(bullet[1]);
      continue;
    }

    para.push(line);
  }
  flushPara();
  flushList();
  flushTable();
  return out.join("\n\n");
}

const source = fs.readFileSync(inputPath, "utf8");
const { attrs, body } = stripFrontMatter(source);
const content = renderMarkdown(body);
const title = attrs.title || "vime + RL-Kernel";
const summary = attrs.summary || "";

const html = `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>${escapeHtml(title)}</title>
</head>
<body style="margin: 0; background: #f5f7fb;">
  <main style="box-sizing: border-box; width: 100%; max-width: 780px; margin: 0 auto; padding: 28px 18px 48px; background: #ffffff; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', Arial, sans-serif;">
    <h1 style="margin: 0 0 14px; color: #0f172a; font-size: 28px; line-height: 1.35; font-weight: 800;">${escapeHtml(title)}</h1>
    ${summary ? `<p style="margin: 0 0 24px; color: #526071; font-size: 15px; line-height: 1.75;">${escapeHtml(summary)}</p>` : ""}
    ${content}
  </main>
</body>
</html>
`;

fs.writeFileSync(outputPath, html, "utf8");
console.log(outputPath);
