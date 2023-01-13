# Gender Classifier
For this quick delivery, I attempted to tackle it by using two approaches with hyperparameter tuning, i.e., machine learning (via PyCaret framework) and deep learning (via tensorflow). With an end-to-end development mindset, I created a web app that leverages the final deep learning model via major cloud providers (i.e., Azure and GCP).

## Deliverable
- Web App
    - [Azure](https://gender-classifier-beta.azurewebsites.net) (stopped)
    - [GCP](https://gender-classifier-n7asbfuu5a-as.a.run.app) (stopped)

- RestAPI
```bash
curl --location --request POST 'https://gender-classifier-n7asbfuu5a-as.a.run.app/api' --header 'Content-Type: application/json' --data-raw '{"First Name": <user input>}'
```
## Usage
## installation (assumed in a virtual env)
```bash
pip install -r requirements.txt
```

## Model Playground
- [ML](code/gender_classifier_ml.ipynb) (final model excluded from repo)
- [DL](code/gender_classifier_dl.ipynb) (final model included in repo)
- [ML-localtest](./local_test.py) `python local_test.py`


## Launch app locally using Gunicorn
```bash
gunicorn -w 2 app:app -b localhost:8000
```

## Docker Build
```bash
docker build --rm -t gender_classifier:v0 . #use buildx build --platform linux/amd64 for mac m1 in order to deploy to cloud
docker run -d -p 8000:8080 gender_classifier:v0
```
