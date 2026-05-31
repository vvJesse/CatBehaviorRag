from __future__ import annotations

from io import BytesIO

from pypdf import PdfReader

from rag_document_uploader.text_normalizer import OCRTextNormalizer


class OCRService:
    """负责 OCR 引擎调用与回退逻辑。"""

    def __init__(self, rapid_ocr=None):
        self._rapid_ocr = rapid_ocr
        self._tesseract_languages: set[str] | None = None

    def _get_rapid_ocr(self):
        """惰性初始化 RapidOCR，首次使用时才导入，避免启动时触发 onnxruntime DLL 加载。"""
        if self._rapid_ocr is None:
            from rapidocr_onnxruntime import RapidOCR  # noqa: PLC0415
            self._rapid_ocr = RapidOCR()
        return self._rapid_ocr

    def recognize(self, image) -> str:
        """优先使用 RapidOCR，必要时回退到 Tesseract。"""
        rapidocr_text = self._extract_rapidocr_text(image)
        if rapidocr_text:
            return rapidocr_text

        return self._extract_tesseract_text(image)

    def _extract_rapidocr_text(self, image) -> str:
        import numpy as np  # noqa: PLC0415
        result, _ = self._get_rapid_ocr()(np.array(image))
        if not result:
            return ""

        lines: list[str] = []
        for item in result:
            if len(item) < 2:
                continue

            text = str(item[1]).strip()
            if text:
                lines.append(text)

        return "\n".join(lines).strip()

    def _extract_tesseract_text(self, image) -> str:
        import pytesseract  # noqa: PLC0415
        languages = self._get_tesseract_languages()

        try:
            if "chi_sim" in languages:
                return pytesseract.image_to_string(image, lang="chi_sim+eng").strip()

            if "eng" in languages:
                return pytesseract.image_to_string(image, lang="eng").strip()
        except pytesseract.TesseractError:
            return ""

        return ""

    def _get_tesseract_languages(self) -> set[str]:
        import pytesseract  # noqa: PLC0415
        if self._tesseract_languages is None:
            try:
                self._tesseract_languages = set(pytesseract.get_languages(config=""))
            except pytesseract.TesseractError:
                self._tesseract_languages = set()

        return self._tesseract_languages


class PDFTextExtractor:
    """负责 PDF 文本层提取和 OCR 回退。"""

    def __init__(
        self,
        ocr_service: OCRService,
        text_normalizer: OCRTextNormalizer,
        ocr_visible_text_threshold: int = 20,
    ):
        self.ocr_service = ocr_service
        self.text_normalizer = text_normalizer
        self.ocr_visible_text_threshold = ocr_visible_text_threshold

    def extract(self, pdf_bytes: bytes) -> str:
        """优先读取 PDF 文本层；必要时再回退到 OCR。"""
        reader = PdfReader(BytesIO(pdf_bytes))
        pages = []

        for page in reader.pages:
            page_text = page.extract_text() or ""
            pages.append(page_text)

        if not any(self._needs_ocr(page_text) for page_text in pages):
            return "\n\n".join(filter(None, pages)).strip()

        return self._extract_with_ocr(pdf_bytes, pages)

    def _needs_ocr(self, page_text: str) -> bool:
        visible_text = "".join(page_text.split())
        return len(visible_text) < self.ocr_visible_text_threshold

    def _extract_with_ocr(self, pdf_bytes: bytes, extracted_pages: list[str]) -> str:
        import pypdfium2 as pdfium  # noqa: PLC0415
        from tqdm import tqdm  # noqa: PLC0415
        pdf = pdfium.PdfDocument(pdf_bytes)
        pages: list[str] = []

        for index, existing_text in tqdm(
            enumerate(extracted_pages),
            total=len(extracted_pages),
            desc="pdf_pages",
            unit="page",
        ):
            if not self._needs_ocr(existing_text):
                pages.append(existing_text.strip())
                continue

            image = pdf[index].render(scale=3).to_pil()
            ocr_text = self.ocr_service.recognize(image)
            pages.append(ocr_text or existing_text)

        pages = [self.text_normalizer.normalize(text) for text in pages]
        return "\n\n".join(filter(None, pages)).strip()


class EPUBTextExtractor:
    """从文字型 EPUB 提取正文，按 spine 顺序保留章节与段落结构，不依赖 OCR。"""

    # 提取这些块级语义标签的文本作为独立段落
    _BLOCK_TAGS = ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "dd", "dt"]
    # 删除这些标签及其内容（导航、样式、脚本、文档头）
    _REMOVE_TAGS = ["style", "script", "nav", "head"]

    def extract(self, epub_bytes: bytes) -> str:
        """从 EPUB 二进制内容提取纯文本，按 spine 顺序拼接各章节。"""
        try:
            import ebooklib
            from ebooklib import epub as _epub
        except ImportError as exc:
            raise ImportError("解析 EPUB 需要安装 ebooklib: pip install ebooklib") from exc
        try:
            from bs4 import BeautifulSoup  # noqa: F401
        except ImportError as exc:
            raise ImportError("解析 EPUB 需要安装 beautifulsoup4: pip install beautifulsoup4") from exc

        book = _epub.read_epub(BytesIO(epub_bytes))
        chapters: list[str] = []

        for item_id, _ in book.spine:
            item = book.get_item_with_id(item_id)
            if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
                continue

            html_content = item.get_content().decode("utf-8", errors="replace")
            chapter_text = self._extract_chapter_text(html_content)
            if chapter_text:
                chapters.append(chapter_text)

        return "\n\n".join(chapters).strip()

    def _extract_chapter_text(self, html_content: str) -> str:
        """将单个 XHTML 文档转换为以空行分隔的段落文本。"""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html_content, "html.parser")

        for tag in soup(self._REMOVE_TAGS):
            tag.decompose()

        paragraphs: list[str] = []
        for tag in soup.find_all(self._BLOCK_TAGS):
            # 跳过已被同类块级标签嵌套的子元素，避免文字重复提取
            if tag.find_parent(self._BLOCK_TAGS):
                continue
            text = tag.get_text(separator="", strip=True)
            if text:
                paragraphs.append(text)

        return "\n\n".join(paragraphs)


class UploadedFileParser:
    """负责按文件类型分发到对应的解析逻辑。"""

    def __init__(
        self,
        support_file_types: list[str],
        pdf_extractor: PDFTextExtractor,
        epub_extractor: EPUBTextExtractor | None = None,
    ):
        self.support_file_types = support_file_types
        self.pdf_extractor = pdf_extractor
        self.epub_extractor = epub_extractor or EPUBTextExtractor()

    def parse(self, file_name: str, file_bytes: bytes) -> str:
        """按文件后缀将二进制内容分发给对应的解析方法。"""
        suffix = file_name.rsplit('.', maxsplit=1)[-1].lower() if '.' in file_name else ''

        if suffix not in self.support_file_types:
            raise ValueError(f"不支持的文件类型: {suffix}")

        if suffix == "txt":
            return self._read_text(file_bytes)

        if suffix == "pdf":
            return self.pdf_extractor.extract(file_bytes)

        if suffix == "epub":
            return self.epub_extractor.extract(file_bytes)

        raise ValueError(f"无法读取的文件类型: {suffix}")

    @staticmethod
    def _read_text(file_bytes: bytes) -> str:
        """将文本文件按 UTF-8 解码为字符串。"""
        return file_bytes.decode("utf-8", errors="replace")