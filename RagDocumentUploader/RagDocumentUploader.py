from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from pypdf import PdfReader
import pypdfium2 as pdfium
import pytesseract
from rapidocr_onnxruntime import RapidOCR
from tqdm import tqdm

import Config


def fix_ocr_text(text):
    """
    修复 OCR 断开的文字：合并断行、保留段落、清理空格
    """
    import re

    # 1. 替换所有单个换行（不是段落分隔）为空格
    # 匹配：非换行符 + 换行 + 非换行符 → 替换为 空格
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

    # 2. 把多个连续换行变成标准分段（保留段落）
    text = re.sub(r'\n+', '\n\n', text)

    # 3. 清理多余空格（多个空格变一个）
    text = re.sub(r' +', ' ', text)

    # 4. 去除首尾空白
    text = text.strip()

    return text


class RagDocumentUploader(object):
    def __init__(self):
        """初始化文件解析器的配置。"""
        self.supportFileType = Config.support_file_types  # now only pdf and txt
        self._rapid_ocr = RapidOCR()
        self._tesseract_languages = None

    def upload_file(self, uploaded_file):
        """读取前端上传对象，并返回解析后的文本内容。"""
        if uploaded_file is None:
            raise ValueError("未收到上传文件")

        # app_update_rag_doc.py 通过 streamlit 的 UploadedFile 调用这里。
        # 这个判断的目的，是在入口处尽早校验上传对象是否具备后续解析所需的两个能力：
        # 1. name: 用于判断文件后缀。
        # 2. getvalue(): 用于拿到二进制内容。
        # 如果缺少其中任何一个属性，后续逻辑会在更深层的位置报错，不利于定位问题。
        if not hasattr(uploaded_file, "name") or not hasattr(uploaded_file, "getvalue"):
            raise ValueError("上传对象格式不正确")

        full_text = self._parse_file_bytes(uploaded_file.name, uploaded_file.getvalue())
        self.save_full_text(uploaded_file, full_text)
        return full_text

        return

    def save_full_text(self, upload_file, full_text):
        """将解析后的全文保存到配置目录，并返回保存后的文件路径。"""
        output_path = self._build_full_text_path(upload_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(full_text, encoding="utf-8")
        return output_path

    @staticmethod
    def _build_full_text_path(upload_file):
        """根据上传文件名生成对应的文本保存路径。"""
        if upload_file is None or not hasattr(upload_file, "name"):
            raise ValueError("无法生成保存路径，上传对象缺少文件名")

        source_name = Path(upload_file.name)
        safe_name = source_name.stem or "uploaded_file"
        return Path(Config.full_text_path) / f"{safe_name}.txt"

    def _parse_file_bytes(self, file_name, file_bytes):
        """按文件后缀将二进制内容分发给对应的解析方法。"""
        suffix = Path(file_name).suffix.lower().lstrip(".")

        if suffix not in self.supportFileType:
            raise ValueError(f"不支持的文件类型: {suffix}")

        if suffix == "txt":
            return self._read_text(file_bytes)

        if suffix == "pdf":
            return self._read_pdf(file_bytes)

        raise ValueError(f"无法读取的文件类型: {suffix}")

    @staticmethod
    def _read_text(file_bytes):
        """将文本文件按 UTF-8 解码为字符串。"""
        return file_bytes.decode("utf-8", errors="replace")

    def _read_pdf(self, pdf_bytes):
        """优先读取 PDF 文本层；必要时再回退到 OCR。"""
        reader = PdfReader(BytesIO(pdf_bytes))
        pages = []

        # reader.pages 中的每一项都是 PDF 页面对象；extract_text() 返回字符串或 None。
        for page in reader.pages:
            page_text = (page.extract_text() or "")
            pages.append(page_text)

        # 如果所有页面都已经有足够的文本层，就直接拼接返回，不进入 OCR 分支。
        if not any(self._needs_ocr(page_text) for page_text in pages):
            return "\n\n".join(filter(None, pages)).strip()

        return self._read_pdf_with_ocr(pdf_bytes, pages)

    @staticmethod
    def _needs_ocr(page_text):
        """根据可见文本长度判断当前页是否需要 OCR。"""
        visible_text = "".join(page_text.split())
        return len(visible_text) < 20

    def _read_pdf_with_ocr(self, pdf_bytes, extracted_pages):
        """仅对疑似扫描页执行 OCR，并复用已有文本层结果。"""
        # pdf 的类型是 pdfium.PdfDocument，用于按页渲染 PDF。
        pdf = pdfium.PdfDocument(pdf_bytes)
        # pages 是 list[str]，用于累积每一页最终保留下来的文本结果。
        pages = []

        for index, existing_text in tqdm(enumerate(extracted_pages), total=len(extracted_pages)):
            # index 是 int，existing_text 是当前页已有的文本层内容 str。
            if not self._needs_ocr(existing_text):
                pages.append(existing_text.strip())
                continue

            # image 是 PIL.Image.Image，对扫描页进行 OCR 前先把 PDF 页面渲染成位图。
            image = pdf[index].render(scale=3).to_pil()
            # ocr_text 是 str，保存 OCR 提取出来的文本。
            ocr_text = self._extract_ocr_text(image)
            pages.append(ocr_text or existing_text)

        pages = [fix_ocr_text(txt) for txt in pages]
        # 最终返回值是 str，把所有非空页面文本拼接为完整文档。
        return "\n\n".join(filter(None, pages)).strip()

    def _extract_ocr_text(self, image):
        """优先使用无需系统中文语言包的 OCR；必要时再退回 Tesseract。"""
        rapidocr_text = self._extract_rapidocr_text(image)
        if rapidocr_text:
            return rapidocr_text

        return self._extract_tesseract_text(image)

    def _extract_rapidocr_text(self, image):
        """使用 RapidOCR 提取中英文文本。"""
        result, _ = self._rapid_ocr(np.array(image))
        if not result:
            return ""

        lines = []
        for item in result:
            if len(item) < 2:
                continue

            text = str(item[1]).strip()
            if text:
                lines.append(text)

        return "\n".join(lines).strip()

    def _extract_tesseract_text(self, image):
        """当 RapidOCR 未识别出内容时，尝试使用本机 Tesseract。"""
        languages = self._get_tesseract_languages()

        try:
            if "chi_sim" in languages:
                return pytesseract.image_to_string(image, lang="chi_sim+eng").strip()

            if "eng" in languages:
                return pytesseract.image_to_string(image, lang="eng").strip()
        except pytesseract.TesseractError:
            return ""

        return ""

    def _get_tesseract_languages(self):
        """缓存当前机器上的 Tesseract 语言包，避免每页重复查询。"""
        if self._tesseract_languages is None:
            try:
                self._tesseract_languages = set(pytesseract.get_languages(config=""))
            except pytesseract.TesseractError:
                self._tesseract_languages = set()

        return self._tesseract_languages


if __name__ == "__main__":
    sample_path = Path(__file__).resolve().parent / "raw_data" / "家有恶猫.pdf"
    # 这里把本地文件包装成和前端上传对象一致的接口，便于直接复用 upload_file。
    debug_upload = BytesIO(sample_path.read_bytes())
    debug_upload.name = sample_path.name

    uploader = RagDocumentUploader()
    result = uploader.upload_file(debug_upload)
    print(result[:500])
