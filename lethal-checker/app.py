from flask import Flask, render_template, request, redirect, url_for, session
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'
app.secret_key = 'your_secret_key'

def load_sa_templates(template_folder="digit_templates"):
    templates = {}
    for i in range(4):
        path = os.path.join(template_folder, f"{i}.png")
        templates[str(i)] = cv2.imread(path, 0)
    ca_path = os.path.join(template_folder, "CA.png")
    if os.path.exists(ca_path):
        templates["CA"] = cv2.imread(ca_path, 0)
    return templates

def recognize_sa_digit(sa_img, templates):
    sa_gray = cv2.cvtColor(sa_img, cv2.COLOR_BGR2GRAY)
    best_score = -1
    best_digit = "0"
    for digit, template in templates.items():
        result = cv2.matchTemplate(sa_gray, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, _ = cv2.minMaxLoc(result)
        if score > best_score:
            best_score = score
            best_digit = digit
    return best_digit  # "0"～"3"または "CA"

def analyze_image(filepath, side, type, pref, position,starter,sa):
    img = cv2.imread(filepath)

    if side =="1P":
        dora=img[113:134,558:895]
        myhp=img[60:99,165:875]
        tekihp=img[60:99,1045:1760]
        sa_img=img[956:1031,97:137]
        n = 21817
        m = 21882
    else:
        dora=img[113:134,1025:1361]
        myhp=img[60:99,1045:1760]
        tekihp=img[60:99,165:875]
        sa_img=img[956:1031,1784:1820]
        n = 21882
        m = 21817

    cv2.imwrite("myhpcut.png",myhp)
    cv2.imwrite("tekihpcut.png",tekihp)
    cv2.imwrite("doracut.png",dora)

    myhpbar = cv2.imread("myhpcut.png")
    tekihpbar = cv2.imread("tekihpcut.png")
    dorabar = cv2.imread("doracut.png")

    myhp_hsv_image = cv2.cvtColor(myhpbar, cv2.COLOR_BGR2HSV)
    tekihp_hsv_image = cv2.cvtColor(tekihpbar, cv2.COLOR_BGR2HSV)
    dora_hsv_image = cv2.cvtColor(dorabar, cv2.COLOR_BGR2HSV) 

    lower_hue = 0  # 下限
    upper_hue = 179 # 上限

    myhp_mask = cv2.inRange(myhp_hsv_image, (lower_hue, 90, 50), (upper_hue, 255, 255))
    tekihp_mask = cv2.inRange(tekihp_hsv_image, (lower_hue, 90, 50), (upper_hue, 255, 255))
    dora_mask = cv2.inRange(dora_hsv_image, (lower_hue, 170, 50), (upper_hue, 255, 255))

    kernel = np.ones((5,5), np.uint8)
    myhp_mask = cv2.morphologyEx(myhp_mask, cv2.MORPH_OPEN, kernel)
    tekihp_mask = cv2.morphologyEx(tekihp_mask, cv2.MORPH_OPEN, kernel)
    dora_mask = cv2.morphologyEx(dora_mask, cv2.MORPH_OPEN, kernel)

    myhp_pixel_sum = np.sum(myhp_mask) #全ピクセルの輝度の合計をpixel_sumに代入
    myhp_white_pixel_number = myhp_pixel_sum/255 #白いピクセルの数を計算しwhite_pixel_numberに代

    tekihp_pixel_sum = np.sum(tekihp_mask) #全ピクセルの輝度の合計をpixel_sumに代入
    tekihp_white_pixel_number = tekihp_pixel_sum/255 #白いピクセルの数を計算しwhite_pixel_numberに代

    dora_pixel_sum = np.sum(dora_mask) #全ピクセルの輝度の合計をpixel_sumに代入
    dora_white_pixel_number = dora_pixel_sum/255 #白いピクセルの数を計算しwhite_pixel_numberに代

    myHP = myhp_white_pixel_number/n*10000
    tekiHP = tekihp_white_pixel_number/m*10000
    dora = dora_white_pixel_number/711

    myHP = round(myHP,1)
    tekiHP = round(tekiHP,1)
    dora = round(dora,1)
    templates = load_sa_templates()
    sa_digit = recognize_sa_digit(sa_img, templates)
    
    if pref=="normal":
        situstion="ノーマル"
    elif pref=="counter":
        situstion="カウンター"
    elif pref=="punish":
        situstion="パニッシュ"
    else:
        situstion="all"


    if position == "1":
        position = 1
    else:
        position = 2

    return {
        "myHP": myHP,
        "tekiHP": tekiHP,
        "drive": dora,
        "sa_digit": sa_digit,
        "type": int(type),
        "situstion": situstion,
        "position": int(position),
        "starter": starter,
        "sa": sa
    }




def load_combos_from_csv(path='comb.csv'):
    df = pd.read_csv(path)
    return df.to_dict(orient='records')

RUK_COMBOS = load_combos_from_csv('comb.csv')

def sa_to_int(sa_str):
    if sa_str == "CA":
        return 4  
    try:
        return int(sa_str)
    except ValueError:
        return 0  

def get_matching_combos(myHP, tekiHP, drive, sa_digit,type, situstion,position,starter,sa):
    if myHP <= 2500 and str(sa_digit) == "3":
        sa_digit = "CA"
    else:
        sa_digit = str(sa_digit)
    combos = []
    sa_digit_val = sa_to_int(sa_digit)
    sa_input_val = sa_to_int(sa) if sa != "all" else sa_digit_val
    max_usable_sa = min(sa_digit_val, sa_input_val)


    for combo in RUK_COMBOS:
        combo_sa_val = sa_to_int(combo["sa"])
        if combo["damage"] < tekiHP:
            continue
        if type != combo["type"] and combo["type"] != 0:
            continue
        if combo["position"] != position and combo["position"] != 0:
            continue
        if situstion != "all" and combo["situstion"] != situstion:
            continue
        if starter != "all" and starter != combo["starter"]:
            continue
        if drive < combo["drive"]:
            continue
        if combo_sa_val > max_usable_sa:
            continue
        combos.append(combo)
    return sorted(combos, key=lambda x: x['damage'], reverse=True)

        

@app.route('/reset', methods=['POST'])
def reset():
    folder = 'static'
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        # 画像ファイルだけ消す（PNG, JPG, JPEGなど）
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                os.remove(filepath)
                print(f"削除: {filepath}")
            except Exception as e:
                print(f"削除失敗: {filepath}, エラー: {e}")
    session.pop('image_path', None)
    session.pop('combos', None)
    session.pop('result', None)
    session.pop('image_url', None)

    return redirect('/')

PER_PAGE = 15

@app.route('/', methods=['GET', 'POST'])
def index():
    page = request.args.get('page', 1, type=int)
    combos = []
    total = 0
    result = None
    image_url = None
    


    if request.method == 'POST':
        if 'image' in request.files and request.files['image'].filename != '':
            file = request.files['image']
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            session['image_path'] = filepath  # セッションに保存
        else:
            # アップロードがないときは前回のファイルを使う
            filepath = session.get('image_path')

        if not filepath or not os.path.exists(filepath):
            return render_template('index.html', result="画像がありません", combos=[], page=page, total=0, per_page=PER_PAGE)
        image_url = filepath
        side = request.form['side']
        type = request.form['type']
        pref = request.form['pref']
        starter = request.form['starter']
        position = request.form['position']
        sa = request.form['sa']
        result_data = analyze_image(filepath, side, type, pref, position, starter, sa)
        combos = get_matching_combos(
            result_data["myHP"], result_data["tekiHP"], result_data["drive"],
            result_data["sa_digit"], result_data["type"], result_data["situstion"],
            result_data["position"], result_data["starter"],result_data["sa"]
        )
        result = (
            f"自分のHP: {result_data['myHP']} / 敵のHP: {result_data['tekiHP']} / "
            f"ドライブゲージ: {result_data['drive']} / SAゲージ: {result_data['sa_digit']}"
        )
        session['combos'] = combos
        session['result'] = result
        session['image_url'] = image_url

    else:
        combos = session.get('combos', [])
        result = session.get('result')
        image_url = session.get('image_url', None)

    total = len(combos)
    start = (page - 1) * PER_PAGE
    end = start + PER_PAGE
    paginated_combos = combos[start:end]

    return render_template('index.html', result=result, image_url=image_url, combos=paginated_combos, page=page, total=total, per_page=PER_PAGE)

if __name__ == '__main__':
    app.run(debug=True)

