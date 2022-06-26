import numpy as np
import cv2
import os
import shutil
from pdf2image import convert_from_path    # should also get poppler (choco install poppler)
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

import requests
import io
import json

START_DIR = os.getcwd()
image_number = 0

lines = 1

# methods definitions
def pgcount_to_string (count):
    if(int(count/10) == 0):
        return str(0) + str(count)
    return str(count)

def page_name(count):
    return "out_" + pgcount_to_string(count) + ".jpg"

def deldir(dir):
    if os.path.exists(dir) and os.path.isdir(dir):
        shutil.rmtree(dir)

def img_to_text():
    # soln_count = 0
    # for filename in os.listdir():
    #     if filename.endswith(".png"):

    img = cv2.imread(filename)
    _, compressedimage = cv2.imencode('.jpg', img, [1, 90])
    file_bytes = io.BytesIO(compressedimage)

    #OCR API 
    url_api = "https://api.ocr.space/parse/image"

    with open('../ocr_api_key', 'r') as file:
        ocr_api_key = file.read().rstrip()

    result = requests.post(url_api, files={"ROI_0.png": file_bytes}, data={"apikey": ocr_api_key})

    result = result.content.decode()

    result = json.loads(result)
    text_detected = result.get("ParsedResults")[0].get("ParsedText")
    print(text_detected)

        # else:
        #     continue
    
    os.chdir("..")

    return 0

def cut_rectangle(page):
    global image_number

    image = cv2.imread(page)
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blurred, 230,255,cv2.THRESH_BINARY_INV)[1]

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    min_area = 10000
    for c in cnts:
        area = cv2.contourArea(c)
        if area > min_area:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
            ROI = original[y:y+h, x:x+w]
            # cv2.imwrite("ROI_{}.png".format(pgcount_to_string(image_number)), ROI)

            text = pytesseract.image_to_string(ROI, lang='eng', config='--psm 3' )

            if len(text.strip(" ")) == 0:
                '''
                 pytesseract is unable to scan this box for some reason...
                 we will send it to OCR API 
                '''
                cv2.imwrite("retry.jpg", ROI)
                img = cv2.imread("retry.jpg")
                _, compressedimage = cv2.imencode('.jpg', img, [1, 90])
                file_bytes = io.BytesIO(compressedimage)

                #OCR API 
                url_api = "https://api.ocr.space/parse/image"

                with open('../ocr_api_key', 'r') as file:
                    ocr_api_key = file.read().rstrip()
                result = requests.post(url_api, files={"username.jpg": file_bytes}, data={"apikey": ocr_api_key})
                result = result.content.decode()
                result = json.loads(result)

                text_detected = result.get("ParsedResults")[0].get("ParsedText")
                text = text_detected

            print("Response " + str(image_number) + ": " + text)
            print("------------------------")

            image_number += 1
    cv2.waitKey(0)

    os.remove(page)
    # os.remove("temp.png")
    return 0

def ROI_edit(img_name):

    image = cv2.imread(img_name)
    original = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([154, 50, 50], dtype="uint8")
    upper = np.array([175, 255, 255], dtype="uint8")
    mask = cv2.inRange(image, lower, upper)
    
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cv2.fillPoly(mask, cnts, (255,255,255))
    mask = cv2.bitwise_not(mask)  # revert mask to original

    colored = original.copy()
    colored[mask == 255] = (255,255,255)
    result = colored

    if cv2.countNonZero(mask) > 1:
        # print("PURPLE IS PRESENT!")
        cv2.imwrite(img_name, result)
        cut_rectangle(img_name)

    cv2.waitKey(0)
    return 0
    

def pdf_to_jpg(pdf_name):
    pages = convert_from_path(pdf_name, 500)

    # creates a directory with pdf name
    pdf_dir = pdf_name.split(".")[0]
    deldir(pdf_dir) # if folder already exists, delete folder
    os.mkdir(pdf_dir)
    os.chdir(pdf_dir)

    count = 1
    for page in pages:
        img_name = page_name(count)
        page.save(img_name, 'JPEG')
        ROI_edit(img_name)
        count += 1
        
    return 0

def init():
    pdf_name = 'lab6.pdf'
    pdf_to_jpg(pdf_name) 
    # img_to_text()

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0

def main():
    init()
    return 0


if __name__=="__main__":
    main()


