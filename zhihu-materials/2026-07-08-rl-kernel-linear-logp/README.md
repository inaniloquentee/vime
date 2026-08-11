# 知乎发布材料包

这是“方案 B”的知乎发布材料：正文保留为 Markdown，复杂宽表格转成图片，普通段落和代码块交给知乎原生编辑器。

## 文件

- `article-zhihu.md`：知乎正文源稿。
- `source-tables.md`：三张表格图片对应的原始 Markdown 表格。
- `assets/01-rlk-global-architecture.png`：架构图。
- `assets/02-rlk-linear-logp-dataflow.png`：linear_logp 数据流图。
- `assets/03-rlk-linear-logp-speedup.png`：CUDA timing 柱状图。
- `assets/04-rlk-linear-logp-memory.png`：显存柱状图。
- `assets/05-table-cuda-timing.png`：CUDA timing 表格图片。
- `assets/06-table-memory.png`：显存表格图片。
- `assets/07-table-health.png`：训练健康指标表格图片。

## 发布步骤

1. 打开 `article-zhihu.md`。
2. 复制正文到知乎编辑器。
3. 逐张上传 `assets/` 里的图片，替换编辑器里的本地图片占位。
4. 代码块用知乎代码块格式再检查一遍。
5. 如果知乎表格支持良好，可以用 `source-tables.md` 里的原始表格替换 `05/06/07` 三张表格图片。

## 注意

- 当前 `01-04` 四张图来自 `wechat-assets` 目录。你前面贴的新版宽图没有作为文件落到本机，所以这里先使用本地现有 PNG。
- 知乎不会完整保留公众号 HTML/CSS。这个材料包的目标是“知乎上可读且尽量接近原文”，不是 HTML 逐像素复刻。
- 宽表格使用图片是为了避免移动端横向滚动和列宽被知乎重排。
