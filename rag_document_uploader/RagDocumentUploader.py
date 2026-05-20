from __future__ import annotations

from io import BytesIO
from pathlib import Path

import Config
from rag_document_uploader.file_parser import OCRService, PDFTextExtractor, UploadedFileParser
from rag_document_uploader.storage import FullTextStorage
from rag_document_uploader.text_normalizer import OCRTextNormalizer


def fix_ocr_text(text):
    """兼容旧调用，委托给独立的 OCR 文本清洗器。"""
    return OCRTextNormalizer().normalize(text)


class RagDocumentUploader(object):
    def __init__(self):
        """初始化文件解析器的配置。"""
        self.supportFileType = Config.support_file_types
        self._text_normalizer = OCRTextNormalizer()
        self._storage = FullTextStorage(Config.full_text_path)
        self._ocr_service = OCRService()
        self._pdf_extractor = PDFTextExtractor(
            ocr_service=self._ocr_service,
            text_normalizer=self._text_normalizer,
        )
        self._file_parser = UploadedFileParser(
            support_file_types=self.supportFileType,
            pdf_extractor=self._pdf_extractor,
        )

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

        full_text = self._file_parser.parse(uploaded_file.name, uploaded_file.getvalue())
        self.save_full_text(uploaded_file, full_text)
        return full_text

    def save_full_text(self, upload_file, full_text):
        """将解析后的全文保存到配置目录，并返回保存后的文件路径。"""
        output_path = self._build_full_text_path(upload_file)
        return self._storage.save(output_path.name, full_text)

    def _build_full_text_path(self, upload_file):
        """根据上传文件名生成对应的文本保存路径。"""
        if upload_file is None or not hasattr(upload_file, "name"):
            raise ValueError("无法生成保存路径，上传对象缺少文件名")

        return self._storage.build_path(upload_file.name)

    def _parse_file_bytes(self, file_name, file_bytes):
        """按文件后缀将二进制内容分发给对应的解析方法。"""
        return self._file_parser.parse(file_name, file_bytes)

    @staticmethod
    def _read_text(file_bytes):
        """将文本文件按 UTF-8 解码为字符串。"""
        return UploadedFileParser._read_text(file_bytes)

    def _read_pdf(self, pdf_bytes):
        """优先读取 PDF 文本层；必要时再回退到 OCR。"""
        return self._pdf_extractor.extract(pdf_bytes)

    @staticmethod
    def _needs_ocr(page_text):
        """根据可见文本长度判断当前页是否需要 OCR。"""
        visible_text = "".join(page_text.split())
        return len(visible_text) < 20

    def _read_pdf_with_ocr(self, pdf_bytes, extracted_pages):
        """仅对疑似扫描页执行 OCR，并复用已有文本层结果。"""
        return self._pdf_extractor._extract_with_ocr(pdf_bytes, extracted_pages)

    def _extract_ocr_text(self, image):
        """优先使用无需系统中文语言包的 OCR；必要时再退回 Tesseract。"""
        return self._ocr_service.recognize(image)

    def _extract_rapidocr_text(self, image):
        """使用 RapidOCR 提取中英文文本。"""
        return self._ocr_service._extract_rapidocr_text(image)

    def _extract_tesseract_text(self, image):
        """当 RapidOCR 未识别出内容时，尝试使用本机 Tesseract。"""
        return self._ocr_service._extract_tesseract_text(image)

    def _get_tesseract_languages(self):
        """缓存当前机器上的 Tesseract 语言包，避免每页重复查询。"""
        return self._ocr_service._get_tesseract_languages()


if __name__ == "__main__":
    sample_path = Path(__file__).resolve().parent / "raw_data" / "家有恶猫.pdf"
    # 这里把本地文件包装成和前端上传对象一致的接口，便于直接复用 upload_file。
    debug_upload = BytesIO(sample_path.read_bytes())
    debug_upload.name = sample_path.name

    uploader = RagDocumentUploader()
    result = uploader.upload_file(debug_upload)
    print(result[:500])
