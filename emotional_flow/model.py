import torch
from torch import nn
from torch.autograd import Variable


class EmotionFlowModel(nn.Module):
    """
    Emotion flow model
    """
    NB_FILTER = 1024
    NB_BLOCKS = 4
    EMOTION_RNN_LAYER_SIZE = 16

    def __init__(self, text_sequence_dim, emotion_sequence_dim, embedding_dim, output_dim, vocab_size,
                 batch_size, emotion_vec_len, class_weights):
        super(EmotionFlowModel, self).__init__()
        self.use_gpu = torch.cuda.is_available()
        self.batch_size = batch_size
        self.emotion_sequence_dim = emotion_sequence_dim
        self.emotion_vec_len = emotion_vec_len

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.class_weights = class_weights
        if self.use_gpu:
            self.class_weights = self.class_weights.cuda()

        # 4 Convolution blocks with filter sizes 2 to 5
        self.conv1 = Conv1dBlock(self.NB_FILTER, 2, embedding_dim, text_sequence_dim)
        self.conv2 = Conv1dBlock(self.NB_FILTER, 3, embedding_dim, text_sequence_dim)
        self.conv3 = Conv1dBlock(self.NB_FILTER, 4, embedding_dim, text_sequence_dim)
        self.conv4 = Conv1dBlock(self.NB_FILTER, 5, embedding_dim, text_sequence_dim)

        # Bi-LSTM to capture the flow of emotions
        self.emotion_rnn = nn.LSTM(emotion_sequence_dim, self.EMOTION_RNN_LAYER_SIZE, batch_first=False)
        self.emotion_rnn_hidden = self.init_emotion_hidden()
        self.relu = nn.ReLU()
        self.emotion_attn = Attention(self.emotion_vec_len)

        self.mlp_layers = nn.Sequential(
            nn.Linear(self.NB_FILTER * self.NB_BLOCKS + self.EMOTION_RNN_LAYER_SIZE, 500),
            nn.Dropout(0.4),
            nn.Linear(500, 200),
            nn.Dropout(0.4)
        )

        self.softmax = nn.Softmax()
        self.output_layer = nn.Linear(200, output_dim)

    def forward(self, X):
        X1 = X[0]
        X2 = X[1]
        embedded_input = self.embedding(X1)

        embedded_input = embedded_input.permute(0, 2, 1)

        conv_output1 = self.conv1(embedded_input)
        conv_output2 = self.conv2(embedded_input)
        conv_output3 = self.conv3(embedded_input)
        conv_output4 = self.conv4(embedded_input)

        conv_output = torch.cat([conv_output1, conv_output2, conv_output3, conv_output4], 1)
        conv_output = conv_output.view(-1, self.num_flat_features(conv_output))

        # Emotion Flow with RNN
        emotion_rnn_output, self.emotion_rnn_hidden = self.emotion_rnn(X2, (self.emotion_rnn_hidden[0].detach(),
                                                                            self.emotion_rnn_hidden[1].detach()))
        emotion_rnn_output = emotion_rnn_output.permute(0, 2, 1)
        emotion_rnn_output = self.relu(emotion_rnn_output)
        pooled_emotion_rnn_output = self.emotion_attn(emotion_rnn_output)

        pooled_emotion_rnn_output = pooled_emotion_rnn_output.squeeze().view(conv_output.shape[0], -1)

        # Concat two outputs
        concatenated_representation = torch.cat([conv_output, pooled_emotion_rnn_output], 1)

        mlp_output = self.mlp_layers(concatenated_representation)

        tag_probabilities = self.output_layer(mlp_output)

        if self.class_weights.shape[-1] == tag_probabilities.shape[-1]:
            tag_probabilities = tag_probabilities * self.class_weights

        tag_probabilities = self.softmax(tag_probabilities)

        return tag_probabilities

    @staticmethod
    def num_flat_features(input_features):
        size = input_features.size()[1:]
        num_features = 1

        for s in size:
            num_features *= s

        return num_features

    def init_emotion_hidden(self):
        hidden = Variable(torch.randn(1, self.emotion_vec_len, self.EMOTION_RNN_LAYER_SIZE))
        cell = Variable(torch.randn(1, self.emotion_vec_len, self.EMOTION_RNN_LAYER_SIZE))

        if self.use_gpu:
            return hidden.cuda(), cell.cuda()
        else:
            return hidden, cell


class Conv1dBlock(nn.Module):
    """
    Basic 1D convolution block containing filters > activation > pooling
    """

    def __init__(self, nb_filter, filter_size, embedding_dim, sequence_length):
        super(Conv1dBlock, self).__init__()

        self.conv = nn.Conv1d(embedding_dim, nb_filter, filter_size)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(sequence_length - filter_size + 1)
        self.attention = Attention(sequence_length - filter_size + 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, embedded_input):
        conv_output = self.conv(embedded_input)
        activated_output = self.relu(conv_output)
        pooled_output = self.pool(activated_output)

        return self.dropout(pooled_output)


class Attention(nn.Module):
    def __init__(self, dimension):
        super(Attention, self).__init__()

        self.u = nn.Linear(dimension, dimension)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(1)

    def forward(self, h):
        x = self.u(h)
        x = self.tanh(x)
        x = self.softmax(x)
        output = x * h
        output = torch.sum(output, dim=2)
        return output
