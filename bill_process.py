"""
Usage python bill_process.py -h
"""

from PIL import Image
from tesserocr import PyTessBaseAPI, RIL
from fuzzywuzzy import process
import re

import argparse

parser = argparse.ArgumentParser(
    description='Extract data from utility bills')
parser.add_argument('document_type', help="document type",
                    choices=['image', 'pdf'], type=str)
parser.add_argument('document_path', help="document path", type=str)
args = parser.parse_args()


image = Image.open(args.document_path)
classes = ['Total Charges from your last bill', 'total new charges', 'electricity charges', 'total amount due', 'energy delivery',
           'demand delivery']
data = {}

with PyTessBaseAPI() as api:
    api.SetImage(image)
    boxes = api.GetComponentImages(RIL.TEXTLINE, True)
    print('Found {} textline image components.'.format(len(boxes)))
    for i, (im, box, _, _) in enumerate(boxes):
        # im is a PIL image object
        # box is a dict with x, y, w and h keys
        api.SetRectangle(box['x'], box['y'], box['w'], box['h'])
        ocrResult = api.GetUTF8Text()
        conf = api.MeanTextConf()

        if len(ocrResult.strip(' ')) != 0:

            text = re.sub(r'([\W0-9]|kWh|kW)+', " ", ocrResult)

            (match, acc) = process.extractOne(text, classes)

            if acc > 95:
                if match == "energy delivery" or match == "demand delivery":
                    amount = re.findall(r"[\d+.,]+\s+kW", ocrResult)
                else:
                    amount = re.findall(r"\$[^\]]+", ocrResult)

                if amount != []:
                    amount = amount[0].strip('\n')

                data[match] = amount

    print("Done")


for key in data:
    print("{}: {}".format(key, data[key]))
