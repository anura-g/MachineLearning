from flask import Flask, render_template, Response
from ML.camera import Webcam

app = Flask(__name__)
video_stream = Webcam()

@app.route('/')
def index():
    return render_template('style_transfer.html')

def generate_image(camera):
    while True:
        try:
            frame = camera.get_frame()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        except: pass

def generate_styled_image(camera):
    while True:
        try:
            frame = camera.style_image()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        except: pass

@app.route('/real_video_feed')
#Return the generated frames
def real_video_feed():
    return Response(generate_image(video_stream),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/style_video_feed')
#Return the generated frames
def style_video_feed():
    return Response(generate_styled_image(video_stream),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=False)