import os
import sys
from datetime import datetime
import cv2
import numpy as np
from flask import Flask, render_template, request, url_for, jsonify, send_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cbam import load_cbam_model
from grade_severity import grade_severity, segment_lesion
from services.treatment import get_treatment
from services.database import (init_db, save_record, get_records,
                               get_stats, get_dashboard_data, get_record_by_id)
from services.report import generate_report

app = Flask(__name__)

MODEL_PATH = r"D:\python\TomatoDiseaseProject\runs\detect\runs\detect\cbam_s\weights\best.pt"
UPLOAD_DIR = os.path.join("static", "uploads")
RESULT_DIR = os.path.join("static", "results")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

init_db()
model = load_cbam_model(MODEL_PATH)

CLASS_CN = {
    "Tomato__BacterialSpot": "细菌性斑点病",
    "Tomato__EarlyBlight": "早疫病",
    "Tomato__Healthy": "健康",
    "Tomato__LateBlight": "晚疫病",
    "Tomato__LeafMold": "叶霉病",
    "Tomato__MosaicVirus": "花叶病毒",
    "Tomato__SeptoriaLeafSpot": "斑枯病",
    "Tomato__SpiderMites": "二斑叶螨",
    "Tomato__TargetSpot": "靶斑病",
    "Tomato__YellowLeafCurlVirus": "黄化曲叶病毒",
}

SEV_CN = {"Healthy": "健康", "Mild": "轻度", "Moderate": "中度", "Severe": "重度"}
SEV_STYLE = {"Healthy": "success", "Mild": "info", "Moderate": "warning", "Severe": "danger"}


@app.route("/")
def index():
    return render_template("index.html", active="home")


@app.route("/detect", methods=["POST"])
def detect():
    f = request.files.get("image")
    if not f or f.filename == "":
        return render_template("index.html", error="请选择图片", active="home")

    fname = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{f.filename}"
    up_path = os.path.join(UPLOAD_DIR, fname)
    f.save(up_path)

    img = cv2.imread(up_path)
    if img is None:
        return render_template("index.html", error="图片读取失败", active="home")

    result = grade_severity(model, img)

    if result.get("status") == "no_detection":
        return render_template("result.html",
                               has_result=False,
                               img_url=url_for("static", filename=f"uploads/{fname}"),
                               message="未检测到病害叶片",
                               active="home")

    res = model.predict(img, conf=0.25, verbose=False)[0]
    best = int(res.boxes.conf.argmax())
    x1, y1, x2, y2 = map(int, res.boxes.xyxy[best])

    anno = img.copy()
    cv2.rectangle(anno, (x1, y1), (x2, y2), (0, 200, 0), 2)
    cv2.putText(anno, f"{result['class']} {result['conf']:.2f}",
                (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    anno_name = f"anno_{fname}"
    cv2.imwrite(os.path.join(RESULT_DIR, anno_name), anno)

    lesion_url = None
    if "healthy" not in result["class"].lower() and not result.get("note"):
        h, w = img.shape[:2]
        crop = img[max(y1 - 10, 0):min(y2 + 10, h), max(x1 - 10, 0):min(x2 + 10, w)]
        lesion, _, _ = segment_lesion(crop, result["class"])
        overlay = crop.copy()
        overlay[lesion > 0] = (0, 0, 255)
        vis = cv2.addWeighted(crop, 0.6, overlay, 0.4, 0)
        lesion_name = f"lesion_{fname}"
        cv2.imwrite(os.path.join(RESULT_DIR, lesion_name), vis)
        lesion_url = url_for("static", filename=f"results/{lesion_name}")

    advice = get_treatment(result["class"], result["severity"])
    cls_cn = CLASS_CN.get(result["class"], result["class"])

    record_id = save_record(
        time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        image_path=fname,
        result_image_path=anno_name,
        cls_en=result["class"],
        cls_cn=cls_cn,
        conf=result["conf"],
        severity=result["severity"],
        ratio=result["ratio"],
        advice=advice,
    )

    return render_template("result.html",
                           has_result=True,
                           img_url=url_for("static", filename=f"uploads/{fname}"),
                           anno_url=url_for("static", filename=f"results/{anno_name}"),
                           lesion_url=lesion_url,
                           cls_en=result["class"],
                           cls_cn=cls_cn,
                           conf=f"{result['conf']:.3f}",
                           ratio=result["ratio"],
                           sev=result["severity"],
                           sev_cn=SEV_CN.get(result["severity"], result["severity"]),
                           sev_style=SEV_STYLE.get(result["severity"], "secondary"),
                           note=result.get("note", ""),
                           advice=advice,
                           t1=0.10, t2=0.30,
                           record_id=record_id,
                           active="home")


@app.route("/history")
def history():
    records = get_records(limit=50)
    return render_template("history.html", records=records, active="history")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", active="dashboard")


@app.route("/api/dashboard_data")
def api_dashboard_data():
    return jsonify(get_dashboard_data())


@app.route("/batch", methods=["GET", "POST"])
def batch():
    if request.method == "GET":
        return render_template("batch.html", active="batch")

    files = request.files.getlist("images")
    if not files or all(f.filename == "" for f in files):
        return render_template("batch.html", active="batch", error="请选择至少一张图片")

    results = []
    for f in files:
        if f.filename == "":
            continue

        fname = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{f.filename}"
        up_path = os.path.join(UPLOAD_DIR, fname)
        f.save(up_path)

        img = cv2.imread(up_path)
        if img is None:
            results.append({"filename": f.filename, "error": "图片读取失败"})
            continue

        result = grade_severity(model, img)

        if result.get("status") == "no_detection":
            results.append({
                "filename": f.filename,
                "image_url": url_for("static", filename=f"uploads/{fname}"),
                "cls_cn": "未检测到病害",
                "conf": "-", "sev_cn": "-", "ratio": "-", "advice": "-",
            })
            continue

        res = model.predict(img, conf=0.25, verbose=False)[0]
        best = int(res.boxes.conf.argmax())
        x1, y1, x2, y2 = map(int, res.boxes.xyxy[best])
        anno = img.copy()
        cv2.rectangle(anno, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(anno, f"{result['class']} {result['conf']:.2f}",
                    (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
        anno_name = f"anno_{fname}"
        cv2.imwrite(os.path.join(RESULT_DIR, anno_name), anno)

        cls_cn = CLASS_CN.get(result["class"], result["class"])
        advice = get_treatment(result["class"], result["severity"])

        save_record(
            time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            image_path=fname,
            result_image_path=anno_name,
            cls_en=result["class"],
            cls_cn=cls_cn,
            conf=result["conf"],
            severity=result["severity"],
            ratio=result["ratio"],
            advice=advice,
        )

        results.append({
            "filename": f.filename,
            "image_url": url_for("static", filename=f"uploads/{fname}"),
            "anno_url": url_for("static", filename=f"results/{anno_name}"),
            "cls_cn": cls_cn,
            "cls_en": result["class"],
            "conf": f"{result['conf']:.3f}",
            "sev": result["severity"],
            "sev_cn": SEV_CN.get(result["severity"], result["severity"]),
            "sev_style": SEV_STYLE.get(result["severity"], "secondary"),
            "ratio": result["ratio"],
            "advice": advice,
        })

    return render_template("batch.html", active="batch", results=results)


@app.route("/about")
def about():
    return render_template("about.html", active="about")


@app.route("/export_pdf/<int:record_id>")
def export_pdf(record_id):
    record = get_record_by_id(record_id)
    if not record:
        return "记录不存在", 404

    (rid, time, image_path, result_image_path, cls_en, cls_cn,
     conf, severity, ratio, advice) = record

    img_path = os.path.abspath(os.path.join(UPLOAD_DIR, image_path))
    anno_path = os.path.abspath(os.path.join(RESULT_DIR, result_image_path))

    lesion_name = f"lesion_{image_path}"
    lesion_path = os.path.abspath(os.path.join(RESULT_DIR, lesion_name))
    if not os.path.exists(lesion_path):
        lesion_path = None

    pdf_dir = os.path.join("static", "reports")
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_name = f"report_{rid}.pdf"
    pdf_path = os.path.abspath(os.path.join(pdf_dir, pdf_name))

    generate_report(
        save_path=pdf_path,
        img_path=img_path,
        anno_path=anno_path,
        lesion_path=lesion_path,
        cls_cn=cls_cn,
        cls_en=cls_en,
        conf=f"{conf:.3f}",
        severity_cn=SEV_CN.get(severity, severity),
        ratio=ratio,
        advice=advice,
    )

    return send_file(pdf_path, as_attachment=True, download_name=f"番茄病害报告_{rid}.pdf")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)