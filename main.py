from flask import Flask, render_template, request, jsonify
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

app = Flask(__name__)

# Set up the image captioning model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_captions', methods=['POST'])
def generate_captions():
    # Get the uploaded image file and number of captions from the request
    image_file = request.files['image']
    num_captions = int(request.form['num_captions'])

    # Save the uploaded image to a temporary file
    tmp_file_path = f'tmp/{image_file.filename}'
    image_file.save(tmp_file_path)

    # Upload the image to Imgur and get the URL
    client_id = '074567709d7e086'
    imgur_url = upload_image_to_imgur(tmp_file_path, client_id)

    # Generate captions for the image using the image captioning model
    raw_image = Image.open(requests.get(imgur_url, stream=True).raw).convert('RGB')

    # Unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt")

    beam_size = num_captions  # Beam search size

    outputs = model.generate(
        **inputs,
        num_beams=beam_size,
        num_return_sequences=num_captions,
        do_sample=False
    )

    captions = [processor.decode(seq, skip_special_tokens=True) for seq in outputs]

    # Create a response object with the generated captions
    response = {
        'captions': captions,
        'imgur_url': imgur_url
    }

    # Return the response as a JSON object
    return jsonify(response)


def upload_image_to_imgur(image_file_path, client_id):
    # Open the image file in binary mode
    with open(image_file_path, 'rb') as f:
        # Make a POST request to the Imgur API to upload the image
        response = requests.post(
            'https://api.imgur.com/3/image',
            headers={'Authorization': f'Client-ID {client_id}'},
            files={'image': f}
        )

    # Extract the URL of the uploaded image from the response
    if response.status_code == 200:
        data = response.json()['data']
        return data['link']
    else:
        return None


if __name__ == '__main__':
    app.run(debug=True)
