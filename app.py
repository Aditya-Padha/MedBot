from Flask import Flask, request, jsonify
from flask_cors import CORS
import chat
import chatx as chatb

app = Flask(__name__)
#CORS(app, resources={r"/chat": {"origins": "http://localhost:4200"}})
CORS(app)


@app.route('/chat', methods=['POST'])
def chatx():
    data = request.data.decode('utf-8')
    res = chat.chatRes(data)
    print(res)
    return jsonify(res)


if __name__ == '__main__':
    app.run(port=5000, debug=True)
