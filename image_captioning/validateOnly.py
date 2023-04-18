import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from .data_loader import get_loader
from .preprocessing.build_vocab import Vocabulary
from .models.encoder_decoder_model import EncoderCNN, DecoderRNN
from .validateEnsemble import validate
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
import click
import yaml

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@click.command()
@click.option(
    "--model_path",
    type=str,
    default="image_captioning/models/",
    help="Path for writing out models",
)
@click.option(
    "--crop_size", type=int, default=224, help="Size for randomly cropping images"
)
@click.option(
    "--vocab_path",
    type=click.Path(exists=True),
    default="image_captioning/models/vocab.pkl",
    help="Path to the tokenized Vocab",
)
@click.option(
    "--image_dir",
    type=click.Path(exists=True),
    default="image_captioning/data/resized2014",
    help="Path to resized images",
)
@click.option(
    "--caption_path",
    type=click.Path(exists=True),
    default="image_captioning/data/annotations/captions_train2014.json",
)
@click.option(
    "--val_image_dir",
    type=click.Path(exists=True),
    default="image_captioning/data/resized_val2014",
    help="Path to resized validation images",
)
@click.option(
    "--val_caption_path",
    type=click.Path(exists=True),
    default="image_captioning/data/annotations/captions_val2014.json",
)
@click.option(
    "--log_step", type=int, default=10, help="Step size for printing log info"
)
@click.option(
    "--save_step",
    type=int,
    default=1000,
    help="Step size for caching/saving trained models",
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    default="image_captioning/configs/default.yaml",
    help="Path to config file",
)
@click.option(
    "--beam",
    type=int,
    default=1,
    help="Beam size, if 1 just greedy",
)
@click.option(
    "--encoder_path",
    type=click.types.Path(exists=True),
    help="Path to trained Encoder",
    default="image_captioning/models/encoder-lr4.ckpt",
)
@click.option(
    "--decoder_path",
    type=click.types.Path(exists=True),
    help="Path to tarined Decoder",
    default="image_captioning/models/decoder-lr4.ckpt",
)
@click.option(
    "--decoder_path2",
    type=click.types.Path(exists=True),
    help="Path to trained decoder 2 for ensemble",
    default="image_captioning/models/decoder-lr2.ckpt",
)
@click.option(
    "--decoder_path3",
    type=click.types.Path(exists=True),
    help="Path to trained Decoder 3 for ensemble",
    default="image_captioning/models/decoder-lr.ckpt",
)
def main(
    model_path,
    crop_size,
    vocab_path,
    image_dir,
    caption_path,
    val_image_dir,
    val_caption_path,
    log_step,
    save_step,
    config,
    beam,
    encoder_path,
    decoder_path,
    decoder_path2,
    decoder_path3,
):
    # Create model directory
    print(model_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Get configs
    with open(config) as fp:
        config = yaml.safe_load(fp)
        config = config["Train"]

    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose(
        [
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    # Load vocabulary wrapper
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    # Get Training Data Loader
    data_loader = get_loader(
        image_dir,
        caption_path,
        vocab,
        transform,
        config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )

    # Get Testing Data Loader
    validation_data_loader = get_loader(
        val_image_dir,
        val_caption_path,
        vocab,
        transforms.ToTensor(),
        batch_size=256,
        shuffle=False,
        num_workers=2,
        validation_mode=True,
    )

    (val_images, val_caption_texts) = next(iter(validation_data_loader))
    val_images = val_images.to(device)

    # Build the models
    encoder = EncoderCNN(
        config["encoder_model"], config["encoder_pretrained"], config["embed_size"]
    ).to(device)
    decoder = DecoderRNN(
        config["embed_size"],
        config["hidden_size"],
        len(vocab),
        config["lstm_layers"],
        vocab_path=vocab_path,
        #lstm=False
    ).to(device)
    decoder2 = DecoderRNN(
        config["embed_size"], config["hidden_size"], len(vocab), config["lstm_layers"], vocab_path=vocab_path
    )
    decoder3 = DecoderRNN(
        config["embed_size"], config["hidden_size"], len(vocab), config["lstm_layers"], vocab_path=vocab_path
    )
    decoder2 = decoder.to(device)
    decoder3 = decoder.to(device)

    print(f"device: {device}")

    # Load the trained model parameters
    try:
        encoder.load_state_dict(
            torch.load(encoder_path, map_location=torch.device("cpu"))
        )
    except RuntimeError as e:
        if e.args[0].startswith("Error(s) in loading state_dict for EncoderCNN"):
            print(
                "The checkpoint file was for a different encoder CNN, thus failed to load"
            )
    decoder.load_state_dict(torch.load(decoder_path, map_location=torch.device("cpu")))
    decoder2.load_state_dict(torch.load(decoder_path2, map_location=torch.device("cpu")))
    decoder3.load_state_dict(torch.load(decoder_path3, map_location=torch.device("cpu")))
   
    

                # Show some metrics
    validate(encoder, decoder, decoder2, decoder3, val_images, val_caption_texts, beam=1)


if __name__ == "__main__":
    main()
