import cv2
import numpy as np
import requests
import io
import json

img = cv2.imread("ROI_10.png")
_, compressedimage = cv2.imencode('.jpg', img, [1, 90])
file_bytes = io.BytesIO(compressedimage)

#OCR API 
url_api = "https://api.ocr.space/parse/image"

with open('ocr_api_key', 'r') as file:
    ocr_api_key = file.read().rstrip()

# print (ocr_api_key)

result = requests.post(url_api, files={"ROI_10.png": file_bytes}, data={"apikey": ocr_api_key})

result = result.content.decode()

result = json.loads(result)
text_detected = result.get("ParsedResults")[0].get("ParsedText")
print(text_detected)


# cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()