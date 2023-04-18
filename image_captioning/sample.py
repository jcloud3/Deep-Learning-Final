import torch
import pickle
from torchvision import transforms
from .preprocessing.build_vocab import Vocabulary
from .models.encoder_decoder_model import EncoderCNN, DecoderRNN
from PIL import Image
import click
from .validate import validate, get_metrics
import nltk

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image(image_path, transform=None):
    image = Image.open(image_path).convert("RGB")
    image = image.resize([224, 224], Image.LANCZOS)

    if transform is not None:
        image = transform(image).unsqueeze(0)

    return image


@click.command()
@click.option(
    "--image", type=str, default="sample_images/testfruit.jpg", help="Input Image"
)
@click.option(
    "--encoder_path",
    type=click.types.Path(exists=True),
    help="Path to trained Encoder",
    default="image_captioning/models/encoder-lr2.ckpt",
)
@click.option(
    "--decoder_path",
    type=click.types.Path(exists=True),
    help="Path to tarined Decoder",
    default="image_captioning/models/decoder-lr2.ckpt",
)
@click.option(
    "--vocab_path",
    type=click.types.Path(exists=True),
    help="Path to pickled vocab",
    default="image_captioning/models/vocab.pkl",
)
@click.option(
    "--encoder_model",
    type=str,
    default="resnet152",
    help="Encoder CNN model (resnet or densenet)",
)
@click.option(
    "--encoder_pretrained",
    type=bool,
    default=True,
    help="Whether the encoder CNN model should come pretrained",
)
@click.option(
    "--embed_size", type=int, default=256, help="Dimension of word Embedding Vectors."
)
@click.option(
    "--hidden_size", type=int, default=512, help="Dimension of LSTM hidden states"
)
@click.option("--num_layers", type=int, default=1, help="Number of layers for LSTM")
@click.option(
    "--expected",
    type=str,
    default="a blond man in front of yellow flowers",
    help="A sentence to compare against the generated",
)
@click.option(
    "--beam_size",
    type=int,
    default=3,
    help="Beam size to use for beam search",
)
def generate(
    image,
    encoder_path,
    decoder_path,
    vocab_path,
    encoder_model,
    encoder_pretrained,
    embed_size,
    hidden_size,
    num_layers,
    expected,
    beam_size,
):
    # Image preprocessing
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    # Load vocabulary wrapper
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    # Build models
    encoder = EncoderCNN(
        encoder_model, encoder_pretrained, embed_size
    ).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(
        embed_size, hidden_size, len(vocab), num_layers, vocab_path=vocab_path
    )
    encoder = encoder.to(device)
    decoder = decoder.to(device)

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

    # Prepare an image
    image = load_image(image, transform)
    image_tensor = image.to(device)

    # Generate an caption from the image
    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    sampled_ids = (
        sampled_ids[0].cpu().numpy()
    )  # (1, max_seq_length) -> (max_seq_length)

    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == "<end>":
            break
    sentence = " ".join(sampled_caption)

    # beam = " ".join(decoder.beam_search(3,feature))

    tokens = nltk.tokenize.word_tokenize(str(expected).lower())
    caption_tokens = []
    caption_tokens.append(vocab("<start>"))
    caption_tokens.extend([vocab(token) for token in tokens])
    caption_tokens.append(vocab("<end>"))
    target = torch.Tensor(caption_tokens)
    beam_predict = torch.empty_like(target)

    beam = decoder.beam_search(
        beam_size, feature)
    target_caption = []
    for word_id in beam:
        word = vocab.idx2word[word_id]
        # if word != "<start>" and word != "<end>" :
        target_caption.append(word)
        if word == "<end>":
            break
    beam_sentence = " ".join(target_caption)

    print()
    print(f"Expected sentence         : {expected}")
    print(f"Generated greedy sentence : {sentence}")
    print(f"Generated Beam sentence   : {beam_sentence}")

    stats = ["Bleu4", "Rouge"]

    print("\nGreedy Stats")
    for stat in stats:
        metric = get_metrics(expected, sentence, validation_type=stat)
        print(f"{stat} : {metric}")

    print("\nBeam Stats")
    for stat in stats:
        metric = get_metrics(expected, beam_sentence, validation_type=stat)
        print(f"{stat} : {metric}")

    return


if __name__ == "__main__":
    generate()
