from flask import Flask, render_template, request, jsonify
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

app = Flask(__name__)

# set up the image captioning model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_captions', methods=['POST'])
def generate_captions():
    # get the uploaded image file and number of captions from the request
    image_file = request.files['image']
    num_captions = int(request.form['num_captions'])

    # save the uploaded image to a temporary file
    tmp_file_path = f'tmp/{image_file.filename}'
    image_file.save(tmp_file_path)

    # upload the image to Imgur and get the URL
    client_id = '074567709d7e086'
    imgur_url = upload_image_to_imgur(tmp_file_path, client_id)

    # generate captions for the image using the image captioning model
    raw_image = Image.open(requests.get(imgur_url, stream=True).raw).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt").to("cuda")

    captions = []
    for i in range(num_captions):
        out = model.generate(
            **inputs,
            max_length=32,
            do_sample=True,
            top_k=50,
            top_p=1.0,
            temperature=1.0
        )
        caption = processor.decode(out[0], skip_special_tokens=True)
        captions.append(caption)

    # create a response object with the generated captions
    response = {
        'captions': captions,
        'imgur_url': imgur_url
    }

    # return the response as a JSON object
    return jsonify(response)

def upload_image_to_imgur(image_file_path, client_id):
    # open the image file in binary mode
    with open(image_file_path, 'rb') as f:
        # make a POST request to the Imgur API to upload the image
        response = requests.post(
            'https://api.imgur.com/3/image',
            headers={'Authorization': f'Client-ID {client_id}'},
            files={'image': f}
        )

    # extract the URL of the uploaded image from the response
    if response.status_code == 200:
        data = response.json()['data']
        return data['link']
    else:
        return None

if __name__ == '__main__':
    app.run(debug=True)