from ignite.metrics import Rouge
from ignite.metrics.nlp import Bleu
import torch


def get_metrics(ground_truth, prediction, validation_type="Bleu4"):
    """
    :param: ground_truth    - str / list of strings
    :param: prediction      - str / list of strings
    :kwarg: validation_type - Validation model to use
    """

    if isinstance(prediction, str):
        prediction = [prediction]

    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]

    assert len(prediction) == len(ground_truth)

    prediction = [entry.split() for entry in prediction]
    ground_truth = [entry.split() for entry in ground_truth]

    for idx in range(len(prediction)):
        if prediction[idx][0] != "<start>":
            prediction[idx].insert(0, "<start>")
        if ground_truth[idx][0] != "<start>":
            ground_truth[idx].insert(0, "<start>")

        if prediction[idx][-1] != "<end>":
            prediction[idx].append("<end>")
        if ground_truth[idx][-1] != "<end>":
            ground_truth[idx].append("<end>")

    if validation_type.startswith("Bleu"):
        # https://pytorch.org/ignite/master/generated/ignite.metrics.Bleu.html#ignite.metrics.Bleu
        if validation_type == "Bleu1":
            ngram = 1
        elif validation_type == "Bleu2":
            ngram = 2
        elif validation_type == "Bleu3":
            ngram = 3
        elif validation_type == "Bleu4":
            ngram = 4
        else:
            return

        model = Bleu(ngram=ngram, smooth="smooth1")

    elif validation_type == "Rouge":
        # https://pytorch.org/ignite/master/generated/ignite.metrics.Rouge.html#ignite.metrics.Rouge
        model = Rouge(variants=["L", 2], multiref="best")

    else:
        raise RuntimeError(f"Invalid Validation Type: {validation_type}")
    # TODO elif validation_type == "RougeL", "RougeN"

    for pred, truth in zip(prediction, ground_truth):
        model.update((pred, truth))
    score = model.compute()

    return score


def validate(encoder, decoder, val_images, val_captions, validation_type="", beam=1):
    """
    :param: encoder
    :param: decoder
    :param: val_captions
    :kwarg: validation_type

    Use this to run a test on your encoder and decoder given tensors of images
    and list of captions.
    """
    with torch.no_grad():
        features = encoder(val_images)
        predictions = decoder.sample(features)
        predictions = decoder.convert_tokens_to_text(predictions)
        if beam>1:
            beams = []
            for feature in features:
                beams.append(decoder.beam_search(beam, feature))
            
        for idx in range(3):
            print()
            print(f"sample ground_truth : {val_captions[idx]}")
            print(f"sample prediction  : {predictions[idx]}")
            if beam>1:
                beam_pred = decoder.beam_translate(beams[idx])
                print(f"Sample beam search prediction : {beam_pred}")
        print()

        if validation_type:
            metric_val = get_metrics(val_captions, predictions, validation_type)
            print(f"{validation_type}: {metric_val}")

        else:
            metrics_list = ["Bleu4", "Bleu3", "Rouge"]
            for metric in metrics_list:
                metric_val = get_metrics(val_captions, predictions, metric)
                print(f"{metric} : {metric_val}")
