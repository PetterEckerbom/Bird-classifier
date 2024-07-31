from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

model = tf.keras.models.load_model('model.keras')

labels = ["arcloo","arcter","arcwar1","atlpuf","banswa","bargoo","barswa","batgod","bearee1","bkbplo","bkhgul","bklkit","bktgod","blackc1","blagro1","blakit1","blared1","blasto1","blawoo1","blkgui","blksco1","blkter","blrwar1","blueth","blutit","bohwax","borowl","brambl","brant","brbsan","brnowl","cangoo","carcro1","casgul2","caster1","citwag","coatit2","cogwar1","colfly1","combuz1","comcha","comchi1","comcra","comcuc","comeid-female","comeid-male","comgol-female","comgol-male","comgre","comhom1","comkin1","comloo","commer-female","commer-male","commoo3","commur","compoc","comqua1","comrav","comred","comred1","comred2","comros","comsan","comshe","comsni","comswi","comter","corbun1","corcra","corplo","cowpig1","crelar1","cretit2","cursan","doveki","dunnoc1","eargre","eaywag1","egygoo","ettwoo1","eubeat1","eucdov","eueowl1","eugori2","eugplo","eugwoo2","euhbuz1","eupfly1","eupowl1","euptit1","eurbla-female","eurbla-male","eurbul","eurcoo","eurcur","eurdot","eurgol","eurgre1","eurhob","eurjac","eurjay1","eurkes-female","eurkes-male","eurlin1","eurmag1","eurnig1","eurnut1","eurnut2","euroys1","eurrob1","eurrol1","eurser1","eursha1","eursis","eurspa1","eursta","eurtre1","eurwar1","eurwig","eurwoo","eurwry","eutdov","eutspa","fieldf","firecr1","gadwal-female","gadwal-male","gargan","garwar1","gbbgul","glagul","gnwtea-female","gnwtea-male","goldcr1","goleag","gragoo","graher1","grcgre1","grebit1","grecor","greegr","gresca-female","gresca-male","gresku1","gretit1","grewar3","grewhi1","grgowl","grnsan","grrwar1","grseag1","grswoo","grypar","grywag","gstlar1","gwfgoo","gyfwoo1","gyrfal","hawfin","hazgro1","hergul","hoopoe","horgre","horlar","houspa-female","houspa-male","ictwar1","jacsni","kenplo1","kineid","laplon","lbbgul","lcspet","legshr2","leseag1","leswhi4","leswoo1","lirplo","litbun","litcra1","litgre1","litgul","litsti","litter1","loeowl","lotduc-female","lotduc-male","lotjae","lottit1","lwfgoo","mallar3","manshe","marsan","martit2","marwar3","meapip1","medgul1","merlin-female","merlin-male","mewgul","misthr1","monhar1","mutswa","nohowl","norful","norgan","norgos1","norhar1","norlap","norpin-female","norpin-male","norsho-female","norsho-male","norshr1","norwhe","ortbun1","osprey-female","osprey-male","palhar1","palwar5","parcro2","parjae","pecsan","perfal-female","perfal-male","pieavo1","pifgoo","pingro-female","pingro-male","pomjae","pursan","razorb","rebfly","rebgoo1","rebmer-female","rebmer-male","rebshr1","recpoc","redcro-female","redcro-male","redkit1","redkno","redpha1","redwin","reebun","refblu","reffal1","rengre","renpha","retloo","retpip","ricpip1","rinouz1","rinphe1","rocpig","rocpip1","rocpta1","rolhaw","rook1","rossta2","rudshe","rudtur","ruff","rusbun","sabgul","sander","santer1","savwar1","sedwar1","sheowl","shttre1","sibjay1","skylar","smew","snobun","snoowl1","sonthr1","sooshe","spocra1","spofly1","spored","steeid","stodov1","stonec4-female","stonec4-male","taibeg1","tawowl1","tawpip1","temsti","tersan","thrnig1","trepip","tufduc-female","tufduc-male","tunswa","twite1","uraowl1","watpip1","watrai1","wemhar1-female","wemhar1-male","wescap1","whbwoo1","whimbr","whinch1","whisto1","whiwag","whoswa","whtdip1","whteag","whwcro","whwsco3","whwter","wilpta","wiltit1","winwre4","wlwwar","woolar1","woosan","woowar","yebloo","yebwar3","yelgul1","yellow2"]

def preprocess_image(image):
    image = image.resize((300, 300))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    try:
        image_data = base64.b64decode(data['image'])
        image = Image.open(BytesIO(image_data))
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)[0]  # Get the first (and only) batch
        top_indices = predictions.argsort()[-5:][::-1]  # Indices of top 5 predictions
        top_predictions = [
            {"label": labels[i], "certainty": float(predictions[i])}
            for i in top_indices
        ]

        return jsonify(top_predictions)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
