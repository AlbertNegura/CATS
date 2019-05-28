import json
import joblib
import re

word_to_idx_path = 'emotional_flow/data/word2index.json'
emotion_lexicon_path = 'emotional_flow/data/emotion_lexicons.pkl'

word_to_idx = json.load(open(word_to_idx_path, 'r'))

emotion_lexicon = joblib.load(emotion_lexicon_path)
emotion_types = ['anger', 'anticipation', 'disgust', 'fear', 'joy',
                 'negative', 'positive', 'sadness', 'surprise', 'trust']


#regex = re.compile('[^a-zA-Z]')

def preprocess(text, sequence_length=1500, emotion_vector_length=20, max_idx=5000):
    text = text.lower()
    #text = regex.sub('', text) # remove all non-alpha numeric characters
    word_list = text.split()

    word_sequence = _to_padded_sequence(word_list, sequence_length, max_idx)
    emotion_vector = _emotion_vector(word_list, chunk_size=emotion_vector_length)

    return word_sequence, emotion_vector


def _to_padded_sequence(word_list, sequence_length, max_idx):
    # translate to word indices
    sequence = []
    for word in word_list:
        idx = word_to_idx.get(word)
        if (idx is not None) and idx < max_idx:
            sequence.append(idx)

    return pad_sequence(sequence, sequence_length)


def pad_sequence(sequence, length):
    seq_len = len(sequence)
    pad_len = abs(length - seq_len)
    padded_sequence = [0] * length

    if seq_len < length:
        padding = [0] * (pad_len)
        padded_sequence = padding + sequence
    elif seq_len > length:
        padded_sequence = sequence[-length:]

    return padded_sequence


def _emotion_vector(word_list, chunk_size=20):
    segment_scores = [[0 for j in range(len(emotion_types))] for i in range(chunk_size)]
    segment_length = round(len(word_list) / chunk_size)

    for i in range(chunk_size):
        start_index = i * segment_length
        end_index = (i + 1) * segment_length

        for token in word_list[start_index:end_index]:
            for emo_idx in range(len(emotion_types)):
                if token in emotion_lexicon[emotion_types[emo_idx]]:
                    segment_scores[i][emo_idx] += 1

        total = sum(segment_scores[i])
        if total > 0:
            for j in range(len(segment_scores[i])):
                segment_scores[i][j] /= total
                segment_scores[i][j] *= 100

    return segment_scores
