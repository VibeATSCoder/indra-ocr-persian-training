"""Shared label, crop and split contracts for the larger OCR experiment."""
import hashlib
import io
import math
import unicodedata
import cv2
import numpy as np
from PIL import Image
from indra_ocr.rtl import normalize, visual_label
from indra_ocr.rec_preproc import crop_line

def clean(text):
    text=unicodedata.normalize('NFC',normalize(str(text))).replace('\\n','\n')
    text=''.join(c for c in text if unicodedata.category(c)!='Cf' or c=='\u200c')
    return ' '.join(text.split())

def key(text):
    text=clean(text).translate(str.maketrans({'ي':'ی','ك':'ک','\u200c':'','ـ':''}))
    return hashlib.sha256(text.encode()).hexdigest()

def canonical(text):
    return clean(text).translate(str.maketrans({'ي':'ی','ك':'ک','\u200c':''})).replace(' ','')

def crop_image(content,box=None):
    gray=np.asarray(Image.open(io.BytesIO(content)).convert('L')) if isinstance(content,bytes) else content
    if box is None:
        if np.median(gray)<110:gray=255-gray
        _,binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        n,labels,stats,_=cv2.connectedComponentsWithStats(binary,8)
        keep=np.zeros(n,np.uint8);keep[1:]=(stats[1:,cv2.CC_STAT_AREA]>=2)
        ys,xs=np.where(keep[labels])
        if len(xs)<3:return None
        box=(int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1)
    return np.rint(crop_line(gray,box,max_width=2048)*255).astype(np.uint8)

def fits(ink,text):
    label=visual_label(clean(text))
    need=len(label)+sum(a==b for a,b in zip(label,label[1:]))
    return ink is not None and 3<=len(label)<=240 and need<=math.ceil(ink.shape[1]/4)
