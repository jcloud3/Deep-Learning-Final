# Deep Learning Final Project: Image Captioning

This project is based off of a pytorch example found here: 

https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning

## Environment
To create base environment. Note that some packages can't be installed using conda so we need to do separate pip install step. Pip install will install pytorch with GPU support but make sure you double check at https://pytorch.org/. You also need to install `punkt` for `nltk` for the tokenizer.
```
make conda_env
conda activate cs7643-fp
pip install -r requirements.txt
python -m nltk.downloader punkt
```

## How to Train
### Preprocessing
Before training, we need to preprocess the captions training data by resizing the images and generating a reference vocab file.

1. Download Training Set. There is a total of 82,784 images totalling 13 GB.
```
cd data/
bash get_data_coco_2014.sh
```

2. Generate Vocabulary
```
python -m image_captioning.preprocessing.build_vocab
```

3. Resize Training Images and Validation Images. Size should be around 1.3 GB.
```
python -m image_captioning.preprocessing.resize

python -m image_captioning.preprocessing.resize \
    --image_dir image_captioning/data/val2014 \
    --output_dir image_captioning/data/resized_val2014
```

### Train
```
python -m image_captioning.train
```


## How to Train in Google Collab
Note: When training in Google Collab, if the data is in drive, there will be a network timeout. Its recommended that you zip the training data then move it to collab's local storage and unzip.

1. Upload repo to Google Drive **without** the training data but including the pickled vocab.
2. zip the resized training data and upload it to drive
    - `zip -r resized2014.zip resized2014`
    - `zip -r resized_val2014.zip resized_val2014`
    - `resized2014.zip`, `vocab.pkl`, `resized_val2014.zip`, and `captions_train2014.json` are saved here for convenience https://drive.google.com/drive/folders/1P5uc1sS3n0oucUl5FqbfSN6OVMGHW5Ie?usp=sharing
    - put `resized2014.zip` and `resized_val2014.zip` in the `image_captioning/data/` folder, put `vocab.pkl` in the `image_captioning/models/` folder and put `captions_train2014.json` in the `image_captioning/data/annotations` folder
3. Open cs7643_fp_driver.ipynb in Google Colaboratory [sic]
10. Follow the Jupyter notebook file until `!python -m image_captioning.train`, which starts the actual training

If you have Google Colab Pro, I think you can leave the notebook running even if you close the browser. However, once the training is done, I think Colab will release all the resources. So add a cell after the running `!python train.py` to move the trained model into persistant storage (Google Drive). The trained models should look something like `decoder-3-2000.ckpt` or `encoder-resnet152-3-2000.ckpt`. In this example, 3 is the epoch number, and 2000 is the iteration number (resnet152 is the encoder CNN model). If you leave the `train.py` arguments as default, you should see files that look like `decoder-5-3000.ckpt` and `encoder-resnet152-5-3000.ckpt`.

There is a notebook in this repo `cs7643_fp_driver.ipynb` that you can upload to collab to help with the setup.

## How to Generate Sample Caption
```
bash test.sh
```
or
```
python -m image_captioning.sample \
    --image "sample_images/rabbit.jpg" \
    --encoder_path "image_captioning/models/encoder-resnet152-3-2000.ckpt" \
    --decoder_path "image_captioning/models/decoder-resnet152-3-2000.ckpt" 
```
You can configure the shell script to point to the trained model and the desired image to caption. If you trained your model in Google Colab, you can download the trained models for the Encoder and the Decoder. Make sure you point to the correct path in `test.sh`. Make sure that you're using the correct `vocab.pkl` as well.

## Troubleshooting

* If you get an error that looks like the snippet below, double check your `models/vocab.pkl`. If you generate the vocab in Collab, sometimes it might not match the `vocab.pkl` that you generate locally, so download your `vocab.pkl` from Collab and use that instead.
```
RuntimeError: Error(s) in loading state_dict for DecoderRNN:
        size mismatch for embed.weight: copying a param with shape torch.Size([9956, 256]) from checkpoint, the shape in current model is torch.Size([9948, 256]).
        size mismatch for linear.weight: copying a param with shape torch.Size([9956, 512]) from checkpoint, the shape in current model is torch.Size([9948, 512]).
        size mismatch for linear.bias: copying a param with shape torch.Size([9956]) from checkpoint, the shape in current model is torch.Size([9948]).
```
