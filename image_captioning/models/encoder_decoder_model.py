import pickle
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from ..preprocessing.build_vocab import Vocabulary
import numpy as np

# The following was borrowed from:
# https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning


class EncoderCNN(nn.Module):
    def __init__(self, model, pretrained, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        if model == "resnet18":
            resnet = models.resnet18(pretrained=pretrained)
        elif model == "resnet34":
            resnet = models.resnet34(pretrained=pretrained)
        elif model == "resnet50":
            resnet = models.resnet50(pretrained=pretrained)
        elif model == "resnet101":
            resnet = models.resnet101(pretrained=pretrained)
        elif model == "resnet152":
            resnet = models.resnet152(pretrained=pretrained)
        elif model == "densenet121":
            resnet = models.densenet121(pretrained=pretrained)
        elif model == "densenet161":
            resnet = models.densenet161(pretrained=pretrained)
        elif model == "densenet169":
            resnet = models.densenet169(pretrained=pretrained)
        elif model == "densenet201":
            resnet = models.densenet201(pretrained=pretrained)
        else:
            return

        if "resnet" in model:
            modules = list(resnet.children())[:-1]  # delete the last fc layer.
            self.resnet = nn.Sequential(*modules)
            self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        else:
            # TODO: figure out why hidden layer is size 1000? Maybe default for densenet?
            self.resnet = resnet
            self.linear = nn.Linear(1000, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class DecoderRNN(nn.Module):
    def __init__(
        self,
        embed_size,
        hidden_size,
        vocab_size,
        num_layers,
        max_seq_length=20,
        vocab_path="image_captioning/models/vocab.pkl",
        search_type="greedy",
        lstm=True,
    ):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        if lstm:
            self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        else:
            self.lstm = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        self.search_type = search_type

        with open(vocab_path, "rb") as fp:
            self.vocab = pickle.load(fp)

    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""

        """ not sure why they're passing in captions. Is it for teacher forcing?"""

        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])

        return outputs

    def convert_tokens_to_text(self, sampled_ids):

        predicted_strings = []
        for sample_idx in range(len(sampled_ids)):
            sampled_numpy = sampled_ids[sample_idx].cpu().numpy()

            sampled_caption = []
            for word_id in sampled_numpy:
                sampled_caption.append(self.vocab.idx2word[word_id])

                if sampled_caption[-1] == "<end>":
                    break

            predicted_strings.append(" ".join(sampled_caption))

        return predicted_strings

    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search"""

        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(
                inputs, states
            )  # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(
                hiddens.squeeze(1)
            )  # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)  # predicted: (batch_size)

            sampled_ids.append(predicted)
            inputs = self.embed(predicted)  # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)  # inputs: (batch_size, 1, embed_size)

        sampled_ids = torch.stack(
            sampled_ids, 1
        )  # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids

    def beam_search(self, k, encoded, states=None):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoded.to(device)
        if (encoded.shape[0]!=1):
            encoded = encoded.unsqueeze(0)
        start = [1]
        # go back to most things not being tensors. Just change s to a tensor on line 136
        start_word = [[start, 0.0]]
        current_word = 1
        # encoded = encoded.unsqueeze(1)
        while len(start_word[0][0]) < self.max_seg_length:  # self.max_seg_length
            
            temp = []
            state = None
            for s in start_word:
                curLength = len(s[0])

                seq = torch.Tensor(s[0])

                sequence = nn.functional.pad(
                    seq, (0, self.max_seg_length - curLength)
                ).reshape(
                    (1, self.max_seg_length)
                )  # sequence of most probable words
                sequence = sequence.int().to(device)

                embeddings = self.embed(sequence)
                embeddings = torch.cat((encoded.unsqueeze(1), embeddings), 1)

                packed = pack_padded_sequence(
                    embeddings, torch.Tensor([self.max_seg_length]), batch_first=True
                )

                hiddens, state = self.lstm(packed, state)
                outputs = self.linear(hiddens[0])
                
                # print(outputs.shape)
                word_preds = np.argsort(outputs.cpu()[current_word].detach().numpy())[
                    -k:
                ]  
                
                # word_preds = word_preds.tolist()
                
                temp_prob = s[1]
                for w in word_preds:
                    next_cap = []
                    prob = temp_prob
                    
                    next_cap = (s[0][:])
                    
                        
                    
                    next_cap.append(w)
                    
                    
                    #if outputs[0][w]>0:
                    #    outprob = torch.log(outputs[0][w])+temp_prob  # assign a probability to each K words4
                    #else:
                    outprob = (outputs[0][w])+temp_prob 
                    if w == 3:
                        outprob = 0
                    if next_cap[-1] == 2 and w ==2:
                        outprob *=0.7
                    temp.append([next_cap, outprob])
                

            start_word = temp

            start_word = sorted(start_word, reverse=False, key=lambda l: l[1])

            current_word += 1
            # Getting the top words
            start_word = start_word[-k:]

        start_word = start_word[-1][0]
        
        return start_word
    def beam_translate(self, tokens):
        target_caption = []
        for word_id in tokens:
            word = self.vocab.idx2word[word_id]
            # if word != "<start>" and word != "<end>" :
            target_caption.append(word)
            if word == "<end>":
                break
        beam_sentence = " ".join(target_caption)
        return beam_sentence
    
