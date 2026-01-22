# 课件图片目录

此目录用于存放 `slides_cn.md` 中使用的图片资源。

## 需要的图片

### 1. `cpu-vs-gpu.png`
- **描述**: CPU 和 GPU 架构对比图
- **用途**: Part 3 - 计算和内存差异
- **推荐来源**:
  - Wikimedia Commons: https://commons.wikimedia.org/wiki/File:Cpu-gpu.svg
  - 或使用课件中的表格形式(已包含)

### 2. `llama2-architecture.png`
- **描述**: Llama2 模型架构图或 Hugging Face 模型卡片截图
- **用途**: Part 2 - LLM vs 传统软件
- **推荐来源**:
  - Hugging Face: https://huggingface.co/meta-llama/Llama-2-7b (截图)
  - The Illustrated Transformer: https://jalammar.github.io/illustrated-transformer/
  - Llama2 论文: https://arxiv.org/abs/2307.09288

### 3. `software-llm-comparison.png` (可选)
- **描述**: 传统软件 vs LLM 流程对比图
- **用途**: Part 1 - LLM 本质也是软件
- **推荐**: 使用课件中的 Mermaid 图表(已包含)

## 如何获取图片

1. **运行下载脚本**:
   ```bash
   cd resources/lesson-0-introduction
   bash download_images.sh
   ```

2. **手动下载**:
   - 查看 `IMAGE_GUIDE.md` 中的详细说明
   - 访问推荐的资源链接
   - 将图片保存到此目录

3. **自己创建**:
   - 使用 Excalidraw: https://excalidraw.com
   - 使用 draw.io: https://app.diagrams.net
   - 使用 Figma: https://figma.com

## 版权说明

所有图片应符合以下条件之一:
- ✅ 公共领域 (Public Domain)
- ✅ Creative Commons 许可 (CC-BY, CC-BY-SA)
- ✅ 自己创作
- ✅ 获得授权使用

## 在 Markdown 中使用

基本语法:
```markdown
![图片描述](images/图片名.png)
```

控制大小:
```markdown
![w:800](images/图片名.png)     # 宽度 800px
![h:400](images/图片名.png)     # 高度 400px
![w:800 h:400](images/图片名.png)  # 指定宽高
```

作为背景:
```markdown
![bg](images/图片名.png)         # 全屏背景
![bg right:40%](images/图片名.png)  # 右侧背景，占40%
```

## 当前状态

- [ ] cpu-vs-gpu.png
- [ ] llama2-architecture.png
- [x] 使用 Mermaid 图表替代流程图 ✅
- [x] 使用文字表格替代 CPU vs GPU 图 ✅

> 💡 当前课件已使用 Mermaid 图表和表格，即使没有图片也可以正常展示！
