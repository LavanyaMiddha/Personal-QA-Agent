from pathlib import Path

import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DataLoader:

    def extract_images(pdf, page, page_num, output_dir):
        image_paths = []

        for idx, img in enumerate(page.get_images(full=True)):
            try:
                xref = img[0]

                image_data = pdf.extract_image(xref)

                image_bytes = image_data["image"]
                ext = image_data["ext"]

                image_path = (
                    output_dir
                    / f"page_{page_num+1}_image_{idx}.{ext}"
                )

                with open(image_path, "wb") as f:
                    f.write(image_bytes)

                image_paths.append(str(image_path))

            except Exception as e:
                print(f"Failed image extraction: {e}")

        return image_paths
    
    def is_valid_table(table) -> bool:
        try:
            df = table.to_pandas()

            rows, cols = df.shape

            if rows < 2:
                return False

            if cols < 2:
                return False

            text_chars = (
                df.fillna("")
                .astype(str)
                .applymap(len)
                .sum()
                .sum()
            )

            avg_chars_per_cell = text_chars / max(rows * cols, 1)

            return avg_chars_per_cell > 3

        except Exception:
            return False

    def load_pdf(self, file_path: str):
        pdf = pymupdf.open(file_path)
        documents = []
        for ipage, page in enumerate(pdf):
            text = page.get_text()
            images = page.get_images()
            tables = page.find_tables()
            if images:
                print(f"Found images! in {file_path} on slide {ipage}")
            if tables:
                print(f"Found tables! in {file_path} on slide {ipage}")
            documents.append(Document(page_content=text, metadata={"source": file_path, "page_number": ipage+1}))
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
    documents = data_loader.load_pdfs("data/pdf_files")
    print(f"Pages loaded: {len(documents)}")
    splits = data_loader.return_splits(documents)
    print(f"Splits created: {len(splits)}")
    for i in range(-1, -3, -1):
        print(f"Split {i}: {splits[i]} (Type: {type(splits[i])})")