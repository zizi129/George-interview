from __future__ import annotations

import io
import re
import zipfile
from functools import lru_cache
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image

ALLOWED_RESUME_SUFFIXES = {".txt", ".pdf", ".docx"}
MAX_RESUME_CHARS = 12000
MIN_RESUME_CHARS = 20
MIN_PDF_DIRECT_TEXT_CHARS = 120
MAX_OCR_PAGES = 6
OCR_RENDER_SCALE = 2.0


def _decode_text_file(filebytes: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
        try:
            return filebytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return filebytes.decode("utf-8", errors="ignore")


def _extract_pdf_text(filebytes: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("当前环境缺少 pypdf，暂时无法解析 PDF 简历。") from exc

    reader = PdfReader(io.BytesIO(filebytes))
    chunks: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            chunks.append(page_text)
    return "\n".join(chunks)


@lru_cache(maxsize=1)
def _get_ocr_engine():
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise RuntimeError("当前环境缺少 rapidocr_onnxruntime，暂时无法进行 OCR 识别。") from exc
    return RapidOCR()


def _ocr_image_array(image_array: np.ndarray) -> str:
    result, _ = _get_ocr_engine()(image_array)
    if not result:
        return ""
    lines: list[str] = []
    for item in result:
        if len(item) >= 2 and str(item[1]).strip():
            lines.append(str(item[1]).strip())
    return "\n".join(lines)


def _extract_pdf_text_via_ocr(filebytes: bytes) -> str:
    try:
        import pymupdf
    except ImportError as exc:
        raise RuntimeError("当前环境缺少 pymupdf，暂时无法对 PDF 做 OCR。") from exc

    doc = pymupdf.open(stream=filebytes, filetype="pdf")
    pages_text: list[str] = []
    page_total = min(len(doc), MAX_OCR_PAGES)
    for page_index in range(page_total):
        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(OCR_RENDER_SCALE, OCR_RENDER_SCALE), alpha=False)
        image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        image_array = np.array(image)
        page_text = _ocr_image_array(image_array)
        if page_text.strip():
            pages_text.append(page_text)
    doc.close()
    return "\n".join(pages_text)


def _extract_docx_text(filebytes: bytes) -> str:
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    try:
        with zipfile.ZipFile(io.BytesIO(filebytes)) as archive:
            xml_bytes = archive.read("word/document.xml")
    except KeyError as exc:
        raise RuntimeError("DOCX 文件缺少正文内容，无法解析。") from exc
    except zipfile.BadZipFile as exc:
        raise RuntimeError("上传的 DOCX 文件已损坏，无法解析。") from exc

    root = ET.fromstring(xml_bytes)
    paragraphs: list[str] = []
    for paragraph in root.iterfind(".//w:p", namespace):
        texts = [node.text or "" for node in paragraph.iterfind(".//w:t", namespace)]
        line = "".join(texts).strip()
        if line:
            paragraphs.append(line)
    return "\n".join(paragraphs)


def normalize_resume_text(text: str) -> str:
    lines = []
    for line in (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        cleaned = re.sub(r"\s+", " ", line).strip()
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines).strip()


def extract_resume_content(filebytes: bytes, filename: str) -> dict:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in ALLOWED_RESUME_SUFFIXES:
        raise RuntimeError("当前仅支持 PDF、DOCX、TXT 格式的简历。")
    if not filebytes:
        raise RuntimeError("上传的简历文件为空。")

    ocr_used = False
    if suffix == ".txt":
        raw_text = _decode_text_file(filebytes)
    elif suffix == ".pdf":
        direct_text = normalize_resume_text(_extract_pdf_text(filebytes))
        raw_text = direct_text
        if len(direct_text) < MIN_PDF_DIRECT_TEXT_CHARS:
            ocr_text = normalize_resume_text(_extract_pdf_text_via_ocr(filebytes))
            if len(ocr_text) > len(direct_text):
                raw_text = ocr_text
                ocr_used = True
    else:
        raw_text = _extract_docx_text(filebytes)

    normalized = normalize_resume_text(raw_text)
    if len(normalized) < MIN_RESUME_CHARS:
        raise RuntimeError("未从简历中提取到足够内容，请检查文件格式或更换文件。")
    return {
        "text": normalized[:MAX_RESUME_CHARS],
        "ocr_used": ocr_used,
    }


def extract_resume_text(filebytes: bytes, filename: str) -> str:
    return extract_resume_content(filebytes, filename)["text"]
