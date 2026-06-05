from pathlib import Path

import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DataLoader:

    def load_pdf(self, file_path: str):
        pdf = pymupdf.open(file_path)
        documents = []
        for ipage, page in enumerate(pdf):
            text = page.get_text()
            documents.append(Document(page_content=text, metadata={"source": file_path, "page_number": ipage}))
        return documents

    def load_pdfs(self, file_directory: str):
        files = list(Path(file_directory).rglob("*.pdf"))
        print(f"Found {len(files)} files:")
        for f in files:
            print(f)
        all_documents = []
        for file_path in files:
            all_documents.extend(self.load_pdf(str(file_path)))
        return all_documents

    def return_splits(self, docs):
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
        )
        all_splits = text_splitter.split_documents(docs)
        return all_splits


if __name__ == "__main__":
    data_loader = DataLoader()
    documents = data_loader.load_pdfs("../data/pdf_files")
    print(f"Pages loaded: {len(documents)}")
    splits = data_loader.return_splits(documents)
    print(f"Splits created: {len(splits)}")
    for i in range(-1, -3, -1):
        print(f"Split {i}: {splits[i]} (Type: {type(splits[i])})")