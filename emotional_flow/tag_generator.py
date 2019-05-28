import torch
import json
from emotional_flow.preprocess import preprocess
from emotional_flow.model import EmotionFlowModel

# Constants
vocab_size = 5000
embedding_dim = 300
text_sequence_dim = 1500
batch_size = 32
max_epochs = 300
emotion_sequence_dim = 10
emotion_sequence_length = 20
target_classes = 71
top_n_list = [1, 3, 5, 8, 10]

class_weights_path = 'emotional_flow/data/class_weights.json'
model_path = 'model/emotional flow model.pth'
idx_to_tag_path = 'emotional_flow/data/index_to_tag.json'

# Load necessary data
class_weights = torch.FloatTensor(json.load(open(class_weights_path, 'r')))
idx_to_tag = json.load(open(idx_to_tag_path, 'r'))

model = None


def get_tags():
    return list(idx_to_tag.values())


def predict(text):  # returns estimated probability distribution over all tags
    word_sequence, emotion_vector = preprocess(text, sequence_length=text_sequence_dim, max_idx=vocab_size)

    word_sequence = torch.LongTensor([word_sequence])
    emotion_vector = torch.FloatTensor([emotion_vector])

    model = _get_model()
    model.eval()
    result = model([word_sequence, emotion_vector]).detach().numpy()
    return result.squeeze()


def get_top_tags(text, n=3):
    distribution = predict(text)
    top_n_indx = distribution.argsort()[-n:][::-1]
    tags = [get_tags()[indx] for indx in top_n_indx]
    return tags


def _get_model():
    global model
    if not model:
        model = _create_model()
        if torch.cuda.is_available():
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(model_path, map_location='cpu')

        model.load_state_dict(checkpoint['model_state'])

    return model


def _create_model():
    """
    Creates and returns the EmotionFlowModel.
    Moves to GPU if found any.
    :return:
    """
    model = EmotionFlowModel(text_sequence_dim, emotion_sequence_dim, embedding_dim, target_classes, vocab_size,
                             batch_size, emotion_sequence_length, class_weights)

    if torch.cuda.is_available():
        model = model.cuda()

    return model


if __name__ == '__main__':
    print(get_top_tags('whatever whatever whatever too long bar whatever some text long enough crashes with some words'))
