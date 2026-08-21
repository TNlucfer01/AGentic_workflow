import pypdf

file_path = "./nke-10k-2023.pdf"
reader = pypdf.PdfReader(file_path)
print(reader.get_fields())
# calc the no of pages
print(reader.get_num_pages())
print(reader.pages[0].extract_text())
