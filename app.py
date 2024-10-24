from flask import Flask, request, jsonify
import pickle
from configparser import ConfigParser
from inference import InternVL
import cv2
import time


app = Flask(__name__)

config_path = 'config.ini'
model = InternVL(config_path)
config = ConfigParser()
config.read(config_path)


@app.route('/health', methods=['GET'])
async def health():
    res = {'status':'ok', 'code':200}
    res = pickle.dumps(res)
    return res


@app.route('/', methods=['POST'])
async def predict():
    try:
        data = request.data
        data = pickle.loads(data)
        image = data['arr']
        bbox = data['bbox']
        try:
            prompt = data['prompt']
        except:
            prompt = config.get('Model', 'prompt')

        labels = []
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        t = str(int(time.time()))

        cv2.imwrite('save_img/ori/'+t+'.jpg', image)

        for box in bbox:
            x_min, y_min, x_max, y_max, score = map(float, box)
            crop = img[int(y_min):int(y_max),int(x_min):int(x_max)]
            label = model.predict(crop, prompt)
            labels.append(label)
            cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            cv2.putText(image, f'{label}: {score:.2f}', (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imwrite('save_img/output/'+t+'.jpg', image)

        output = {'labels': labels, 'code': 200}

        res = pickle.dumps(output)
        return res
    except Exception as e: 
        res = {'status':str(e), 'code':300}
        res = pickle.dumps(res)

        return res
    

@app.route('/one_image', methods=['POST'])
async def predict_one_image():
    try:
        data = request.data
        data = pickle.loads(data)
        image = data['arr']
        try:
            prompt = data['prompt']
        except:
            prompt = config.get('Model', 'prompt')

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = model.predict(img, prompt)

        output = {'label': label, 'code': 200}

        res = pickle.dumps(output)
        return res
    except Exception as e: 
        res = {'status':str(e), 'code':300}
        res = pickle.dumps(res)

        return res
    

if __name__ == '__main__':
    ip = config.get('Server', 'ip')
    port = config.get('Server', 'port')
    app.config['JSON_AS_ASCII'] = False
    app.run(host=ip, port=port, threaded=False)
