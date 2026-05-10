import streamlit as st

from RagDocumentUploader.RagDocumentUploader import RagDocumentUploader


def main() -> None:
	"""提供文件上传入口，并在页面上展示解析后的文本内容。"""
	st.set_page_config(page_title="Cat Behavior RAG", page_icon="🐱", layout="centered")

	st.title("Cat Behavior RAG")
	st.write("上传一个文件，然后在页面里直接读取内容。")

	uploaded_file = st.file_uploader("选择一个 txt 或 pdf 文件", type=["txt", "pdf"])

	if uploaded_file is None:
		st.info("请先上传文件。")
		return

	st.success(f"已选择文件：{uploaded_file.name}")

	uploader = RagDocumentUploader()
	try:
		content = uploader.upload_file(uploaded_file)
	except ValueError as exc:
		st.error(str(exc))
		return

	st.write(content)



if __name__ == "__main__":
	main()