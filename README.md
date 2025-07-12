# RAG

## PDF to Markdown

### Grobid

使用 GROBID 转换 PDF 为 txt

full

```bash
docker run --rm --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.8.2
```

minimal

```bash
docker run --rm --init --ulimit core=0 -p 8070:8070 lfoppiano/grobid:latest-full
```

### [MinerU](https://mineru.net/)

```bash
uv add mineru
```

```bash
mineru -p <input_path> -o <output_path>
```
