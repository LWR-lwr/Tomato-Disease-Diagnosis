import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))


def generate_report(save_path, img_path, anno_path, lesion_path,
                    cls_cn, cls_en, conf, severity_cn, ratio, advice):
    c = canvas.Canvas(save_path, pagesize=A4)
    width, height = A4

    c.setFont('STSong-Light', 20)
    c.drawCentredString(width / 2, height - 2 * cm, '番茄病害检测报告')

    c.setFont('STSong-Light', 10)
    c.drawString(2 * cm, height - 3 * cm, f'检测时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    y = height - 4 * cm
    img_width = 5 * cm
    img_height = 5 * cm

    if img_path and os.path.exists(img_path):
        c.drawImage(ImageReader(img_path), 2 * cm, y - img_height,
                    width=img_width, height=img_height, preserveAspectRatio=True)
        c.setFont('STSong-Light', 9)
        c.drawCentredString(2 * cm + img_width / 2, y - img_height - 0.5 * cm, '原图')

    if anno_path and os.path.exists(anno_path):
        c.drawImage(ImageReader(anno_path), 7.5 * cm, y - img_height,
                    width=img_width, height=img_height, preserveAspectRatio=True)
        c.setFont('STSong-Light', 9)
        c.drawCentredString(7.5 * cm + img_width / 2, y - img_height - 0.5 * cm, '检测结果')

    if lesion_path and os.path.exists(lesion_path):
        c.drawImage(ImageReader(lesion_path), 13 * cm, y - img_height,
                    width=img_width, height=img_height, preserveAspectRatio=True)
        c.setFont('STSong-Light', 9)
        c.drawCentredString(13 * cm + img_width / 2, y - img_height - 0.5 * cm, '病征分割')

    y = y - img_height - 2 * cm
    c.setFont('STSong-Light', 13)
    c.drawString(2 * cm, y, '检测信息')
    c.line(2 * cm, y - 0.2 * cm, width - 2 * cm, y - 0.2 * cm)

    y -= 1 * cm
    c.setFont('STSong-Light', 11)
    c.drawString(2 * cm, y, f'病害类别：{cls_cn}（{cls_en}）')
    y -= 0.8 * cm
    c.drawString(2 * cm, y, f'置信度：{conf}')
    y -= 0.8 * cm
    c.drawString(2 * cm, y, f'严重程度：{severity_cn}')
    y -= 0.8 * cm
    if ratio not in ('0.0', '-', 0.0):
        c.drawString(2 * cm, y, f'病征覆盖率：{ratio}')
        y -= 0.8 * cm

    y -= 0.5 * cm
    c.setFont('STSong-Light', 13)
    c.drawString(2 * cm, y, '防治建议')
    c.line(2 * cm, y - 0.2 * cm, width - 2 * cm, y - 0.2 * cm)

    y -= 1 * cm
    c.setFont('STSong-Light', 11)
    text = advice or '暂无建议'
    max_width = width - 4 * cm
    line = ''
    for char in text:
        if c.stringWidth(line + char, 'STSong-Light', 11) < max_width:
            line += char
        else:
            c.drawString(2 * cm, y, line)
            y -= 0.6 * cm
            line = char
    if line:
        c.drawString(2 * cm, y, line)

    c.setFont('STSong-Light', 8)
    c.drawCentredString(width / 2, 1.5 * cm, '本报告由番茄病害检测与分级系统自动生成，仅供参考')

    c.save()