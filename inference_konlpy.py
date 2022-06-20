# ====================== Library ===========================
from unittest.util import _MAX_LENGTH
import pandas as pd
import re
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from transformers import PreTrainedTokenizerFast,GPT2LMHeadModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from konlpy.tag import Mecab

# ==========================================================

# ====================Global Variable=======================
data = pd.read_csv('./data/2.Textranked.csv')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GPT_tok = PreTrainedTokenizerFast.from_pretrained("./models/koGPT2_tokenizer", bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')
GPT = GPT2LMHeadModel.from_pretrained('./models/koGPT2_finetuned')

Q_TKN = "<usr>"
A_TKN = "<sys>"
BOS = '</s>'
EOS = '</s>'
MASK = '<mask>'
PAD = '<pad>'
# ============================================================

# ==========================Similarity========================
def get_df_token(data):
    tagger = Mecab()
    q_pos = []
    for i in range(len(data)):
        q_pos.append(tagger.pos(data['Question'][i]))
    data['tokens'] = q_pos
    pos = ["NNG","NNP","VV","VA"]
    data['tokens'] = data['tokens'].apply(lambda x: [t for (t, p) in x if p in pos])
    return data

def get_ques_token(question):
    tagger = Mecab()
    ques_token = tagger.pos(question)
    ques_token = ques_remove(ques_token)
    return ques_token

def ques_remove(question) :
    pos = ["NNG","NNP", "VV","VA"]
    ques_token = [[t for (t, p) in question if p in pos]]
    return ques_token

def jaccard_similarity(data, question):

    union = set(data).union(set(question))
    intersection = set(data).intersection(set(question))
    jaccard_sim = len(intersection) / len(union)  

    return jaccard_sim

def jaccard_high(data, question, num):
    # data: 데이터프레임, question: 입력한 텍스트(질문), num: 자카드 유사도 상위 갯수
    data['jaccard_similarity'] = data['tokens'].apply(lambda x: jaccard_similarity(x, question[0]))
    return data[['Question', 'Answer', 'tokens', 'jaccard_similarity']].sort_values(['jaccard_similarity'], ascending=False)[:num]

def tokenized_output(tokens):
    return tokens

def cos_similarity(data, question):
    tfidf_vectorizer = TfidfVectorizer(analyzer='word',
                                       tokenizer=tokenized_output,
                                       preprocessor=tokenized_output,
                                       token_pattern=None)
    
    tfidf_data = tfidf_vectorizer.fit_transform(data['tokens'])
    tfidf_question = tfidf_vectorizer.transform(question)

    data['cosine_similarity'] = cosine_similarity(tfidf_data, tfidf_question).reshape(-1, )

    return data[['Question', 'Answer', 'tokens', 'jaccard_similarity', 'cosine_similarity']].sort_values(['cosine_similarity'], ascending=False)
# =======================================================================

# =================================Seq2Seq===============================
class QNA:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1:"EOS"}
        self.n_words = 2
        self.stopword = set([('있', 'VV'), ('하', 'VV'), ('되', 'VV') ])

    def addSentence(self, sentence):
        for word in sentence.split('.'):
            self.addWord(word)

    def addWord(self, sen):
        for word in [ val for (val, nn) in tagger.pos(sen) if (val, nn) not in self.stopword and nn in ('NNG', 'NNP', 'VV', 'VA')]:
            if word not in self.word2index:             
                self.word2index[word] = self.n_words
                self.word2count[word] = 1
                self.index2word[self.n_words] = word
                self.n_words += 1
            else:
              self.word2count[word] += 1

def normalizeString(s):
    s = re.sub(r"([.!?])", r" ", s)
    s = re.sub(r"[^가-힣.!?]+", r" ", s)
    return s

def readchat(question, answer, reverse=False):
    print("Reading lines...")

    lines = open('/content/drive/MyDrive/Textranked.csv', encoding='utf-8').read().strip().split('\n')[1:]
    
    pairs = [[normalizeString(s) for s in l.split(',')] for l in lines]

    if reverse:
        paris = [list(reversed(p)) for p in pairs]
        input_q = QNA(answer)
        output_a = QNA(question)
    else:
        input_q = QNA(question)
        output_a = QNA(answer)
    
    return input_q, output_a, pairs

def filterPair(p):
    return len(p[0].split(' ')) < 2000 and len(p[1].split(' ')) < 2000

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(question, answer, reverse=False):
    input_q, output_a, pairs = readchat(question, answer, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    for pair in pairs:
        input_q.addSentence(pair[0])
        output_a.addSentence(pair[1])
    print("Counted words:")
    print(input_q.name, input_q.n_words)
    print(output_a.name, output_a.n_words)
    return input_q, output_a, pairs

input_q, output_a, pairs = prepareData('question', 'answer', True)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)
              
def indexesFromSentence(QNA, sentence):
    tagger = Mecab()
    word2index_list = []
    stopword = set([('있', 'VV'), ('하', 'VV'), ('되', 'VV') ])
    for sen in sentence.split('.'):
        for word in [ val for (val, nn) in tagger.pos(sen) if (val, nn) not in stopword and nn in ('NNG', 'NNP', 'VV', 'VA')]:
            word2index_list.append(QNA.word2index[word])
    return word2index_list

def tensorFromSentence(QNA, sentence):
    indexes = indexesFromSentence(QNA, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=DEVICE).view(-1, 1)

def evaluate(encoder, decoder, sentence, max_length=200):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_q, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=DEVICE)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=DEVICE)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_a.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

# ========================================================================================

def make_answer(data):
    q = input('Question:')
    q_token = get_ques_token(q)

    data = pd.read_csv('./data/2.Textranked.csv')
    d_token = get_df_token(data)

    result = jaccard_high(data, q_token, 100)
    result = cos_similarity(result, q_token)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Q_TKN = "<usr>"
    A_TKN = "<sys>"
    BOS = '</s>'
    EOS = '</s>'
    MASK = '<mask>'
    PAD = '<pad>'

    if result['cosine_similarity'].iloc[0] >= 0.6:
        return result['Answer'].iloc[0]
    else:
        if len(q) < 200:
            SOS_token = 0
            EOS_token = 1
            MAX_LENGTH = 2000
            tagger = Mecab()

            input_q, output_a, pairs = prepareData('question', 'answer', True)
            encoder = torch.load('../models/Seq2Seq_mecab_encoder1_model')
            encoder.load_state_dict(torch.load('../models/Seq2Seq_mecab_encoder1'))
            decoder = AttnDecoderRNN(256, 7755, dropout_p=0.1)
            decoder.load_state_dict(torch.load('../models/Seq2Seq_mecab_attn_decoder1'))
            evaluate(encoder, decoder, q)
        else:
            GPT_tok = PreTrainedTokenizerFast.from_pretrained("./models/koGPT2_tokenizer", bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')
            GPT = GPT2LMHeadModel.from_pretrained('./models/koGPT2_finetuned')

            with torch.no_grad():
                q = q.strip()
                a = ""
                while 1:
                    input_ids = torch.LongTensor(GPT_tok.encode(Q_TKN + q + A_TKN + a)).unsqueeze(dim=0).to(DEVICE)
                    pred = GPT(input_ids)
                    pred = pred.logits.to('cpu')
                    gen = GPT_tok.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().numpy().tolist())[-1]
                    if gen == EOS:
                        break
                    a += gen.replace("▁", " ")
                print("Answer > {}".format(a.strip()))