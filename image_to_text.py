import sys,os,glob,fitz
from pathlib import Path
from subprocess import call
from pdfminer.high_level import extract_text
from PIL import Image
import pyocr
import pdf2image

# 現在のディレクトリ取得
dirname = os.getcwd()

# pdf出力用フォルダ
output_image = dirname + "\output_pdf\images"
poppler_path = "C:\Program Files\poppler-23.01.0\Library\\bin"

images_path_list = glob.glob(output_image+"\*.jpg")

#Tesseractのインストール場所をOSに教える
tesseract_path = "C:\Program Files\Tesseract-OCR"
if tesseract_path not in os.environ["PATH"].split(os.pathsep):
    os.environ["PATH"] += os.pathsep + tesseract_path

# pyocrが利用するOCRエンジンをTesseractに設定
tools = pyocr.get_available_tools() 
if len(tools) == 0:
    print("OCRエンジンが指定されていません")
    sys.exit(1)
else:
    tool = tools[0]

# OCR(文字認識)させたいPDFをjpg(画像)へ変換
builder=pyocr.builders.TextBuilder(tesseract_layout=6)
for image_path in images_path_list:
    image = Image.open(image_path)
    # image.show()
    result = tool.image_to_string(image,lang="jpn+eng",builder=builder).replace(" ", "").replace("\t", "").strip()
    print(result)
    sys.exit(1)