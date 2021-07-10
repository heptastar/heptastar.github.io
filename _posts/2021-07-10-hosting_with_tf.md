# Serve Trained TF Model Using TensorFlow Serving
>An absolutely beginner's guide.

It seems an easy step to move from training to deployment, but it is usually more complicated than it first seems. The reason is, at least for me, it is very easy to get confused by the tools and their purposes at the beginnign. And it takes sometime to navigate your way back on the right track. So I want to set the stage before the show, and let's talk about what's the purpose. 

As a researcher or data scientist, one of the most common case could be training a non-trivial deep learning model on the local GPU. Then you want to host it for others or yourself to use. If this is the case, and you want to deploy the saved model as a RESTful API for others to call as needed, then TensorFlow Serving would be a great option. It's better than using Flask, because you don't need to write separate scripts for each model. The best part after using it for a while, is that if you use Docker, it's as simple as one command. 

## 1. About Docker
I have a Windows 10 machine, and it's the Home version. It's not really convenient to use Docker as I found out after spending sometime. So I installed Ubuntu as a second operating system on my disk, and use Docker there. If you have to use Windows, the latest Docker would support Windows Pro. There is an older version you can run on Windows 10 Home, but I could not get it work with TensorFlow Serving properly. 
To install Docker, check out [Docker](https://www.docker.com/products/docker-desktop) website. The following commands are all executed in Ubuntu 20.04 LTS. It should work on Macs as well. 

## 2. Install TensorFlow Serving with Docker
It's one command. 
```bash
sudo Docker pull tensorflow/serving
```

## 3. Directory and File
The trained model should be saved under the model's directory, which only have different versions of the model. It's the way TensorFlow expects to see how your models are organized. For my case, I have all my sentiment analysis models under the folder called "sentiment_models". And there are three different trained versions of the model. Each versoin is under a seperate folder. It looks like this, 

```
sentiment_models/
├── 00001
│   ├── assets
│   ├── saved_model.pb
│   └── variables
│       ├── variables.data-00000-of-00001
│       └── variables.index
├── 00002
│   ├── assets
│   ├── saved_model.pb
│   └── variables
│       ├── variables.data-00000-of-00002
│       ├── variables.data-00001-of-00002
│       └── variables.index
└── 00003
    ├── assets
    ├── saved_model.pb
    └── variables
        ├── variables.data-00000-of-00002
        ├── variables.data-00001-of-00002
        └── variables.index
```
This is what happened automatically during model training, when I try to save model version to your model directory "sentiment_models". Each time you save a version, it creates a sub-directory (I change the name to 00001, 00002, 00003) to keep the model graph, assets, and varialbes. So when it's time to serving the model, all you need to do is to point to the model directory (not any of the sub-directories), and it will pick the latest model automatically. 

## 4. Serve the Model
Simply use the following command, and you're good to go. 
```bash
sudo docker run -p 8501:8501 \
--mount type=bind,source=/home/sentiment_models/,\
target=/models/sentiment_models \
-e MODEL_NAME=sentiment_models \
-t tensorflow/serving
```
I was not familiar with Docker when I first used this, so there was a lot of trial and error. If this is your case, be careful with the usage of space. There is a space following any command option, like "--mount", but there is no space after that until the option is complete. The backslash "\" is used to wrap a long command to a new line.

"source" is used to point to the model directory.
"target" is used for request. 

You will see information from the terminal window showing it's up and running. 

## 5. Request through Python
There are two steps: prepare your data, and request for the model prediction. This is how I request to analysis the sentiment for a sentence. Pay attention to the http address, "v1" is by default, and "models/sentiment_models" is what I specified earlier in the command as "target". "predict" is TensorFlow default signatureDef handle. 

```python
import json
data = json.dumps({"signature_name": "serving_default", 
  "instances": ["After two days of comparison, I finally decided to buy this handbag. The color looks even better than the picture!"]})

import requests
headers = {"content-type": "application/json"}

# signatureDef handle is "predict"
json_response = requests.post('http://localhost:8501/v1/models/sentiment_models:predict', data=data, headers=headers)
predictions = json.loads(json_response.text)['predictions']
```
## 6. Request using Curl Command
You can also write a scripts using curl command to request prediction. It will return a JSON. You can access it with key "predictions". For my case, it returns the sentiment scores of the list of sentences I sent. 

```bash
curl -d '{"instances": ["After two days of comparison, I finally bought this handbag. The color is even better than it looks in the picture!"]}' -X POST http://localhost:8501/v1/models/sentiment_models:predict

# returns the following
{
    "predictions": [[0.97756505]]
}
```
It is easy to see that the http address points to the "target" specified in previous docker command, same as calling through python. 

## Summary
In this article, I briefly explained how to serve a trained model using Docker and TensorFlow Serving. It is very easy to setup and run this way. The important take away is that organzing the model property as TensorFlow desires is very important. And if it's your first time to use Docker, be careful with the command convention. 
Happy serving!
