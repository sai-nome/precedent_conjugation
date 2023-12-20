import sys,os,glob,fitz,shutil
from pathlib import Path
from subprocess import call
from pdfminer.high_level import extract_text
from PIL import Image
import pyocr
import pdf2image
import pytesseract
# from pdfminer.layout import LAParams, LTTextBox, LTImage
# from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
# from pdfminer.converter import PDFPageAggregator
# from pdfminer.pdfpage import PDFPage
# from pdfminer.pdfparser import PDFParser
# from pdfminer.pdfdocument import PDFDocument

# 現在のディレクトリ取得
dirname = os.getcwd()
# pdf出力用フォルダ
output_pdf = dirname + "\output_pdf"
poppler_path = "C:\Program Files\poppler-23.01.0\Library\\bin"
output_image = output_pdf + "\images"

#移動用ファイルパス
images_pdf = output_pdf + "\images_pdf"
archive_pdf = output_pdf + "\\archive"

file_path_list = glob.glob(archive_pdf+"\*.pdf")
# print(pytesseract.get_languages())

for pdf_path in file_path_list:
    text = extract_text(pdf_path)
    text_rep = text.replace(" ", "").replace("\t", "").strip()
        
    print(pdf_path,len(text_rep))
    f = open("pdf_text_len.txt", "a")
    text_list = [pdf_path, ": ", str(len(text_rep)), "\n"]
    f.writelines(text_list)
    f.close()