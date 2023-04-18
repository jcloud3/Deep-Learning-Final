import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from .data_loader import get_loader
from .preprocessing.build_vocab import Vocabulary
from .models.encoder_decoder_model import EncoderCNN, DecoderRNN
from .validate import validate
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
    beam
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

    ## DELETE THIS LATER
    # encoder_path = "/home/workdir/cs7643-fp/image_captioning/models/encoder-3-2000.ckpt"
    # decoder_path = "/home/workdir/cs7643-fp/image_captioning/models/decoder-3-2000.ckpt"
    # encoder.load_state_dict(torch.load(encoder_path, map_location=torch.device("cpu")))
    # decoder.load_state_dict(torch.load(decoder_path, map_location=torch.device("cpu")))

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = (
        list(decoder.parameters())
        + list(encoder.linear.parameters())
        + list(encoder.bn.parameters())
    )
    optimizer = torch.optim.Adam(params, lr=config["learning_rate"])

    # Train the models
    total_step = len(data_loader)
    stats = np.zeros((1,4))
    for epoch in range(config["num_epochs"]):
        for step_idx, (images, captions, lengths) in enumerate(data_loader):

            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            # Forward, backward and optimize
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            # Print log info
            if step_idx % log_step == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}".format(
                        epoch,
                        config["num_epochs"],
                        step_idx,
                        total_step,
                        loss.item(),
                        np.exp(loss.item()),
                    )
                )
                stats = np.append(stats,[[epoch,step_idx,loss.item(),np.exp(loss.item())]],axis=0)

            # Save the model checkpoints
            if (step_idx + 1) % save_step == 0:
                torch.save(
                    decoder.state_dict(),
                    os.path.join(
                        model_path, "decoder-{}-{}.ckpt".format(epoch + 1, step_idx + 1)
                    ),
                )
                torch.save(
                    encoder.state_dict(),
                    os.path.join(
                        model_path,
                        "encoder-{}-{}-{}.ckpt".format(
                            config["encoder_model"], epoch + 1, step_idx + 1
                        ),
                    ),
                )
                np.save(os.path.join(model_path,"stats.npy"),stats)

                # Show some metrics
                validate(encoder, decoder, val_images, val_caption_texts, beam=beam)


if __name__ == "__main__":
    main()
