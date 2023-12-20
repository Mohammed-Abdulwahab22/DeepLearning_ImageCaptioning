import os
import pickle
import numpy as np

from tqdm import tqdm
import tensorflow as tf

BASE_DIR = "Your base directory where you put the dataset"
WORKING_DIR = "your working dir that you want to save file like .pkl in"


model = tf.keras.applications.vgg16.VGG16()
#restructring the model
model = tf.keras.models.Model(inputs=model.inputs, outputs=model.layers[-2].output)
#summarize 
#print(model.summary())

#Extracting features from the images
features={}
directory = os.path.join(BASE_DIR, 'Images')

for img_name in tqdm(os.listdir(directory)):
    #load the image from the file
    img_path = directory + '/' + img_name
    image = tf.keras.preprocessing.image.load_img(img_path, target_size=(224,224))
    #convert image pixels to numpy array
    image = tf.keras.preprocessing.image.img_to_array(image)
    #reshape data for model
    image = image.reshape((1, image.shape[0], image.shape[1],image.shape[2]))
    #preprocessing the image for vgg16
    image = tf.keras.applications.vgg16.preprocess_input(image)
    #extract features
    feature = model.predict(image,verbose=0)
    #get image ID
    image_id = img_name.split('.')[0]
    #store feature
    features[image_id] = feature
    
    
#store features in pickle
pickle.dump(features, open(os.path.join(WORKING_DIR, 'features.pkl'),'wb'))

#load features from pickle
with open(os.path.join(WORKING_DIR, 'features.pkl'), 'rb') as f:
          features = pickle.load(f)
          
#Load the caption data
with open(os.path.join(BASE_DIR, 'captions.txt'), 'r') as f:
    next(f)
    captions_doc = f.read()


#create mapping of the image captions
mapping = {}
#process lines
for line in tqdm(captions_doc.split('\n')):
    #split the line by comma (,)
    tokens = line.split(',')
    if len(line) < 2 : 
        continue
    image_id, caption = tokens[0], tokens[1]
    #remove extension from image ID
    image_id = image_id.split('.')[0]
    #convert caption list to string
    caption = " ".join(caption)
    #create list if needed
    if image_id not in mapping:
        mapping[image_id] = []
    #store the caption    
    mapping[image_id].append(caption)

print(len(mapping))

#Preprocess Text Data
def clean(mapping):
    for key , captions in mapping.items():
        for i in range(len(captions)):
            #take on caption at a time
            caption = captions[i]
            #preprocess steps
            #convert to lower case
            caption = caption.lower()
            #delete digits and special chars, etc..
            caption = caption.replace('[^A-Za-z]','')
            #delete additional space
            caption = caption.replace('\s+',' ')
            #add start and end tags to the caption
            caption = '<start> ' + " ".join([word for word in caption.split() if len(word) > 1 ]) + ' <end>'
            captions[i] = caption
            
#preprocess the text
clean(mapping)


all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)
        
#tokenize the text
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1

#get maximum length of the caption available 
max_length = max(len(caption.split()) for caption in all_captions)
print(max_length)


#Train test split
image_ids = list(mapping.keys())
split = int(len(image_ids) * 0.90)
train = image_ids[:split]
test = image_ids[split:]


# startseq girl going into wooden building endseq
#        X                   y
# startseq                   girl
# startseq girl              going
# startseq girl going        into
# ...........
# startseq girl going into wooden building      endseq
# create data generator to get data in batch (avoids session crash)
def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
    # loop over images
    X1, X2, y = list(), list(), list()
    n = 0
    while 1:
        for key in data_keys:
            n += 1
            captions = mapping[key]
            # process each caption
            for caption in captions:
                # encode the sequence
                seq = tokenizer.texts_to_sequences([caption])[0]
                # split the sequence into X, y pairs
                for i in range(1, len(seq)):
                    # split into input and output pairs
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = tf.keras.preprocessing.sequence.pad_sequences([in_seq], maxlen=max_length)[0]
                    # encode output sequence
                    out_seq = tf.keras.utils.to_categorical([out_seq], num_classes=vocab_size)[0]
                    
                    # store the sequences
                    X1.append(features[key][0])
                    X2.append(in_seq)
                    y.append(out_seq)
            if n == batch_size:
                X1, X2, y = np.array(X1), np.array(X2), np.array(y)
                yield [X1, X2], y
                X1, X2, y = list(), list(), list()
                n = 0
                

#Model Creation

##Encoder Model
#image feature model
inputs1 = tf.keras.layers.Input(shape=(4096,))
fe1 = tf.keras.layers.Dropout(0.4)(inputs1)
fe2 = tf.keras.layers.Dense(256,activation='relu')(fe1)

#sequence feature layers
inputs2 = tf.keras.layers.Input(shape=(max_length,))
se1 = tf.keras.layers.Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = tf.keras.layers.Dropout(0.4)(se1)
se3 = tf.keras.layers.LSTM(256)(se2)



#Decoder Model
decoder1 = tf.keras.layers.add([fe2,se3])
decoder2 = tf.keras.layers.Dense(256, activation='relu')(decoder1)
outputs = tf.keras.layers.Dense(vocab_size, activation='softmax')(decoder2)



model = tf.keras.Model(inputs=[inputs1,inputs2], outputs=outputs)
model.compile(loss = 'categorical_crossentropy', optimizer='adam')

#Plot the model
tf.keras.utils.plot_model(model, show_shapes=True)



#train the model
epochs = 15
batch_size = 64
steps = len(train)//batch_size

for i in range(epochs):
    #create the data generator
    generator=data_generator(train, mapping, features, tokenizer, max_length, vocab_size, batch_size)
    model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)
    
    
    
#save the model
model.save(WORKING_DIR +'/ImgCaptioningModel.h5')




#generate captions for images
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

#generate caption for an image
def predict_caption(model, image, tokenizer, max_length):
    #add start tag for generation proccess
    in_text = '<start>'
    #iterate over the max length of sequence 
    for i in range(max_length):
        #encode input sequence
        sequence = tf.keras.preprocessing.text.Tokenizer.texts_to_sequences([in_text])[0]
        #pad the sequence 
        sequence = tf.keras.preprocessing.sequence.pad_sequences([sequence], max_length)
        #predict next word
        yhat = model.predict([image,sequence], verbose=0)
        #get index with high propability 
        yhat = np.argmax(yhat)
        #convert index to word
        word = idx_to_word(yhat, tokenizer)
        #stop if word not found
        if word in None:
            break
        #append word as input for generating next word 
        in_text += " " + word
        #stop if we reach end tag
        if word == '<end>':
            break
    return in_text


#validate with test data
actual, predicted = list(), list()

for key in tqdm(test):
    #get actual caption
    captions =  mapping[key]
    #predict the caption for image 
    y_pred = predict_caption(model, features[key], tokenizer, max_length)
    #split into words
    actual_caption = [caption.split() for caption in captions]
    y_pred = y_pred.split()
    #append to the list 
    actual.append(actual_caption)
    predicted.append(y_pred)
    
from nltk.translate.bleu_score import corpus_bleu
#calculate BLUE score
print("BLUE-1: %f" % corpus_bleu(actual, predicted, weights=(1.0,0,0,0)))
print("BLUE-2: %f" % corpus_bleu(actual, predicted, weights=(0.5,0.5,0,0)))


#visualize the results 
from PIL import Image
import matplotlib.pyplot as plt
def generate_caption(image_name):
    #load the image 
    image_name = "insert image name"
    image_id = image_name.split('.')[0]
    img_path = os.path.join(BASE_DIR, "Images",image_name)
    image = Image.open(img_path)
    captions = mapping[image_id]
    print('---------------actual------------')
    for caption in captions:
        print(caption)
        
    #predict the caption
    y_pred = predict_caption(model, features[image_id], tokenizer, max_length)
    print('--------------------predicted-------------------')
    print(y_pred)
    plt.imshow(image)




    
