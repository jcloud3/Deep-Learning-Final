""""
build_vocab.py

The purpose of this preprocessing step is to tokenize the vocab that appears
in the training captions json.
"""

import nltk
import pickle
from collections import Counter
from pycocotools.coco import COCO
import tqdm
import click


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(json, threshold):
    """Build a simple vocabulary wrapper."""
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    for id in tqdm.tqdm(ids, desc="Tokenizing the Captions"):
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

@click.command()
@click.option("--captions_path", required=False, default="image_captioning/data/annotations/captions_train2014.json")
@click.option("--pickled_output_path", required=False, default="image_captioning/models/vocab.pkl")
@click.option("--threshold", required=False, type=int, default=4)
def cli(captions_path, pickled_output_path, threshold):
    
    vocab = build_vocab(json=captions_path, threshold=threshold)
    
    with open(pickled_output_path, "wb") as fp:
        pickle.dump(vocab, fp)
    
    print(f"Total vocab size: {len(vocab)}")
    print(f"Exported vocab to {pickled_output_path}")

if __name__ == '__main__':
    cli()
