import joblib as joblib
import pandas as pd
import nltk
from nltk.tokenize import TweetTokenizer as TweetTokenizer
from nltk.chunk import tree2conlltags
from preprocessor.preprocess import *
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.dicts.emoticons import emoticons
from textblob import TextBlob
import corenlp
import collections
import spacy
import re
import emoji
import regex
from spacy.tokenizer import Tokenizer
import gensim
import random as rn
import numpy as np
import pprint


#Load Dataset
#Link for downloading contents: https://mega.nz/#F!0kYGRYwK!tGEZ8c5pPdfJe8OwpGrDyg

dataset = pd.read_csv( 'dev.txt',delimiter='\t')
tweets = dataset['turn1']+" "+dataset['turn2']+" "+dataset['turn3']
tw_ = tweets
tok = list()

#Final encoding matrix
matrix3D = np.zeros((2755,50,638))


google_300 = gensim.models.KeyedVectors.load_word2vec_format( "../word2vec/embedings/google_w2v_300.bin" , binary=True )
generics_100 = gensim.models.KeyedVectors.load_word2vec_format( "../word2vec/embedings/generics_w2v_neutral_300.bin" , binary=True , unicode_errors='ignore')

pp = {
    'CC':1,
    'CD' :2,
    'DT' : 3,
    'EX' : 4,
    'FW' : 5,
    'IN' : 6,
    'JJ' : 7,
    'JJR' : 8,
    'JJS' : 9,
    'LS' : 10,
    'MD' : 11,
    'NN' : 12,
    'NNS' : 13,
    'NNP' : 14,
    'NNPS' : 15,
    'PDT' : 16,
    'POS' : 17,
    'PRP' : 18,
    'PRP$' : 19,
    'RB' : 20,
    'RBR' : 21,
    'RBS' : 22,
    'RP' : 23,
    'TO' : 24,
    'UH' : 25,
    'VB' : 26,
    'VBD' : 27,
    'VBG' : 28,
    'VBN' : 29,
    'VBP' : 30,
    'VBZ' : 31,
    'WDT' : 32,
    'WP' : 33,
    'WP$' : 34,
    'WRB' : 35,
    '.': 36,
    ',' : 37,
    ':' : 38}

iob = {
    'B-NP':1,
    'I-NP':2,
    'O':0
}

ner = {
        'PERSON':1,
        'NORP':2,
        'FAC':3,
        'ORG':4,
        'GPE':5,
        'LOC':6,
        'PRODUCT':7,
        'EVENT':8,
        'WORK_OF_ART':9,
        'LAW':10,
        'LANGUAGE':11,
        'DATE':12,
        'TIME':13,
        'PERCENT':14,
        'MONEY':15,
        'QUANTITY':16,
        'ORDINAL':17,
        'CARDINAL':18,
        '':19
}

def countSent(tokens):
    countVPos = 0
    countPos = 0
    countNeutral = 0
    countNeg = 0
    countVNeg = 0
    for core_tok in tokens:
        # Sentiment
        core_w_sent = core_tok.sentiment
        # print(core_w_sent)
        polw = 0
        if core_w_sent == 'Very negative':
            polw = -2
            countVNeg += 1
        if core_w_sent == 'Negative':
            polw = -1
            countNeg += 1
        if core_w_sent == 'Positive':
            polw = 1
            countPos += 1
        if core_w_sent == 'Very positive':
            polw = 2
            countVPos += 1
        if core_w_sent == 'Neutral':
            polw = 0
            countNeutral += 1
    return countVPos, countPos, countNeutral, countNeg, countVNeg


def countEsclamation(tw):
    esclamation = 0
    for token in tw.split():
        if (token == '!' or token == '!!' or token == '!!!' or token == '!!!!'):
            esclamation += 1
    return esclamation

def countQMark(tw):
    esclamation = 0
    for token in tw.split():
        if (token == '!' or token == '!!' or token == '!!!' or token == '!!!!'):
            esclamation += 1
    return esclamation

def countStopWord(tw):
    f = pd.read_fwf ( 'stopword_en' )
    stopList = f[ : ]
    isStop = 0
    for token in tw.split():
        if token in stopList:
            isStop += 1
    return isStop

def countInDictionary(tw):
    f = pd.read_fwf ( 'dict_basic_en' )
    stopList = f[ : ]
    isStop = 0
    for token in tw.split():
        if token in stopList:
            isStop += 1
    return isStop


prefix_re = re.compile(r'''^[\[\("']''')
suffix_re = re.compile(r'''[\]\)"']$''')
infix_re = re.compile(r'''[-~]''')
simple_url_re = re.compile(r'''^https?://''')
def custom_tokenizer(nlp):
    return Tokenizer(nlp.vocab, prefix_search=prefix_re.search,
                                suffix_search=suffix_re.search,
                                infix_finditer=infix_re.finditer,
                                token_match=simple_url_re.match)

def countNouns(tw):
    count = 0
    for tupla in tw:
        if tupla[1] == 'NN' or tupla[1] == 'NNP' or tupla[1] == 'NNS' or tupla[1] == 'NNPS':
            count+=1
    return count

def countVerbs( tw ):
    count = 0
    for tupla in tw:
        if tupla[ 1 ] == 'VB' or tupla[ 1 ] == 'VBD' or tupla[ 1 ] == 'VBG' or tupla[ 1 ] == 'VBN' or tupla[ 1 ] == 'VBN' or tupla[ 1 ] == 'VBP' or tupla[ 1 ] == 'VBZ ':
           count += 1
    return count

def countAdjectivies ( tw ):
    count = 0
    for tupla in tw:
       if tupla[ 1 ] == 'JJ' or tupla[ 1 ] == 'JJR' or tupla[ 1 ] == 'JJS':
          count += 1
    return count

def countPronoun ( tw ):
    count = 0
    for tupla in tw:
        if tupla[ 1 ] == 'PRP' or tupla[ 1 ] == 'PRP$':
            count += 1
    return count

def countAdverb ( tw ):
    count = 0
    for tupla in tw:
        if tupla[ 1 ] == 'RB' or tupla[ 1 ] == 'RBR' or tupla[ 1 ] == 'RBS':
            count += 1
    return count

def countEmoji(text):
    emoji_counter = 0
    data = regex.findall(r'\X', text)
    for word in data:
        if any(char in emoji.UNICODE_EMOJI for char in word):
            emoji_counter += 1
            # Remove from the given text the emojis
            text = text.replace(word, '')

    words_counter = len(text.split())

    return emoji_counter, words_counter

def countHashtags(tw):
    pat = re.compile ( r"#(\w+)" )
    l = pat.findall (tw)
    return len(l)

def numSpecial_meta(tw):
    countMention = 0
    countReserved = 0
    countUrls = 0
    countNumbers= 0
    countPercents = 0
    countEmails = 0
    countMoney = 0
    countPhone = 0
    countTime = 0
    countDate = 0

    for c in tw:
        if c == '_MENTION_':
            countMention+=1
        if c == '_RESERVED_':
            countReserved+=1
        if c == '_URL_' or c == '<url>':
            countUrls+=1
        if c == '<number>':
            countNumbers+=1
        if c == '<percent>':
            countPercents+=1
        if c == '<email>':
            countEmails+=1
        if c == '<money>':
            countMoney+=1
        if c == '<phone>':
            countPhone+=1
        if c == '<time>':
            countTime+=1
        if c == '<date>':
            countDate=1

    return countMention,countReserved,countUrls,countNumbers,countEmails,countMoney,countPhone,countTime,countDate

def checkRetweet(tw):
    if 'RT ' in tw:
        return 1
    else:
        return 0

def countWhitespaces(tw):
    numberWhite = sum ( 1 for c in tw if c == ' ' or c=='   ' )
    return numberWhite

def pecentUpper(tw):
    numberUpper = sum ( 1 for c in tw if c.isupper ( ) )
    wordLength = sum ( 1 for c in tw)
    upperPercent = numberUpper / wordLength
    return upperPercent

def percRepeatedChars(tw):
    totRepetitions = 0;
    d = collections.defaultdict( int )
    for c in tw:
        d[ c ] += 1
    for c in sorted ( d , key=d.get , reverse=True ):
        if d[ c ] > 1:
            totRepetitions = totRepetitions + d[ c ]
    wordLength = sum ( 1 for c in tw )
    repPercent = totRepetitions / wordLength
    return repPercent

client = corenlp.CoreNLPClient(start_server=False, annotators=["tokenize", "ssplit", "pos", "lemma", "ner","sentiment"])








#BEGIN
#Preprocessing and Tokenization
tk = TweetTokenizer()
p = Preprocess()
text_processor = TextPreProcessor (
    # terms that will be normalized
    normalize=[ 'email' , 'percent' , 'money' , 'phone' ,
                'time' , 'url' , 'date' , 'number' ] ,
    fix_html=True ,  # fix HTML tokens
    segmenter="twitter" ,
    corrector="twitter" ,
    unpack_hashtags=True ,  # perform word segmentation on hashtags
    unpack_contractions=True ,  # Unpack contractions (can't -> can not)
    spell_correct_elong=True ,  # spell correction for elongated words
    dicts=[ emoticons ]
)


#REPLACE mentions and replaced chars with RESERVED TAGS
for i in tw_:
    line = p.preprocess_mentions ( i , repl='<mention>' )
    line = p.preprocess_reserved_words ( line , repl='<reserved>' )
    line = text_processor.pre_process_doc(line)
    tok.append ( tk.tokenize ( line ) )


indexs = 0
for line in tok:

    # VECTOR OF EACH SENTENCE
    sentence2D = np.zeros (50)

    original_tw = tweets[indexs]
    reconstructed_tw = ''

    for tok in line:
       reconstructed_tw = reconstructed_tw +' '+bytes(tok, 'utf-8').decode('utf-8','ignore')


    #Percent UpperCases sentence
    sent_upper_pec = pecentUpper(original_tw)

    #Percentage RepeatedChar sentence
    sent_repPercent = percRepeatedChars(original_tw)

    #SentLenght in char
    numChar = sum( 1 for c in original_tw)

    #SentLenght in tokens
    numTok = len(line)

    #numSpecial tokens
    countMention , countReserved , countUrls , countNumbers , countEmails , countMoney , countPhone , countTime , countDate = numSpecial_meta(line)

    per_numbers = countNumbers / numTok
    per_emails = countEmails / numTok
    per_money = countMoney / numTok
    per_phone = countPhone / numTok
    per_time = countTime / numTok
    per_date = countDate / numTok

    #% of emoticons
    percEmoticons = countEmoji(original_tw)[0] / numTok

    #POS tagger tokens
    tagged = nltk.pos_tag(line)
    pattern = 'NP: {<DT>?<JJ>*<NN>}'
    cp = nltk.RegexpParser(pattern)
    cs = cp.parse(tagged)
    iob_tagged = tree2conlltags(cs)
    #pprint(iob_tagged[0][1])

    #Perc of names
    perc_nouns = countNouns(tagged) / numTok

    #Perc of verbs
    perc_verbs = countVerbs ( tagged ) / numTok

    #Perc adjectivities
    perc_adj = countAdjectivies ( tagged ) / numTok

    #Perc pronoum
    perc_pron = countPronoun ( tagged ) / numTok

    #Perc adverbs
    perc_adv = countAdverb ( tagged ) / numTok

    #Perc whitespaces
    percWhite= countWhitespaces(original_tw)/numTok

    #PercEscl
    percEscl = countEsclamation(original_tw) /numTok

    #PercQuest
    percQmark = countQMark ( original_tw ) / numTok

    #PercStop
    percStop = countStopWord( original_tw ) / numTok

    #PercDic
    percDic = countInDictionary( original_tw ) / numTok

    #Chunker and Name Entities using SPACY
    nlp = spacy.load('en')
    nlp.tokenizer = custom_tokenizer ( nlp )
    doc = nlp(reconstructed_tw)
    entTok = [(X, X.ent_iob_, X.ent_type_) for X in doc]

    #CoreNlp Analysis
    text =reconstructed_tw
    ann = client.annotate(text)
    sentences = ann.sentence
    pol = 0
    stokens = []
    cc = 0
    for sent in sentences:
        spol = 0
        if sent.sentiment == 'Very negative':
            spol = -2
        if sent.sentiment == 'Negative':
            spol = -1
        if sent.sentiment == 'Positive':
            spol = 1
        if sent.sentiment == 'Very positive':
            spol = 2
        pol += spol
        if cc == 0:
            stokens = sent.token
        else:
            stokens.MergeFrom(sent.token)
        cc +=1

    #Sentiment sentence
    pol = pol / cc

    #Tokens CoreNlp
    lista_tok = stokens
    countVPos , countPos , countNeutral , countNeg , countVNeg = countSent(lista_tok)

    #Perc VPos
    percVpos = countVPos / numTok

    #Perc Pos
    percPos = countPos / numTok

    # Perc Neutral
    percNeutral = countNeutral / numTok

    # Perc Neg
    percNeg = countNeg / numTok

    # Perc VNeg
    percVNeg = countVNeg / numTok


    #Cicle on each token
    tokIndex = 0
    coreNlpTokIndex = 0
    for token in line:
        #Take iob_tagged pos
        tokpos = iob_tagged[tokIndex][1]
        tokiob = iob_tagged[tokIndex][2]

        w_pos = 39
        try:
            w_pos = pp[tokpos]
        except:
            w_pos = 39

        w_iob = iob[tokiob]

        #ner
        w_ner=ner[entTok[tokIndex+1][2]]


        #Take Entity Type
        try:
            if token in lista_tok[coreNlpTokIndex]:
                core_tok =lista_tok[tokIndex]
                core_w_sent = core_tok.sentiment
                coreNlpTokIndex += 1
        except:
            core_w_sent = 'Neutral'

        #Sentiment
        core_w_sent = core_tok.sentiment
        #print(core_w_sent)
        polw = 0
        if core_w_sent == 'Very negative':
            polw = -2
        if core_w_sent == 'Negative':
            polw = -1
        if core_w_sent == 'Positive':
            polw = 1
        if core_w_sent == 'Very positive':
            polw = 2
        if core_w_sent == 'Neutral':
            polw = 0


        #Percentage Upper
        upperPercent = pecentUpper(token)

        #Percentage RepeatedChar
        repPercent = percRepeatedChars(token)

        #Polarity and Intensity
        token_low = token.lower()
        a = TextBlob(token_low).sentiment
        polarity = a[0]
        intensity = a[1]

        #isEsclamationMark
        esclamation =0
        if (token == '!' or token == '!!' or token == '!!!' or token == '!!!!'):
            esclamation = 1


        # isQuestionMark
        question = 0
        if (token == '?' or token == '??' or token == '???' or token == '????'):
            question = 1


        #isaStopword
        import pickle
        stopList = []
        isStop = 0
        f = pd.read_fwf ( 'stopword_en' )
        stopList = f[:]

        if token_low in stopList:
            isStop = 1


        #isInDictionary
        dictList = [ ]
        isDic = 0
        m = pd.read_fwf ( 'dict_basic_en' )
        dictList = m[ : ]
        if token_low in dictList:
            isDic = 1

        # Take from Embeddings
        g_vec = [ ]
        is_in_model = False
        # token = token.lower()
        if token == "_MENTION_":
            g_vec = google_300.wv[ "mention" ]
            is_in_model = True
        elif token == "_URL_":
            is_in_model = True
            g_vec = google_300.wv[ "url" ]
        elif token == "_RESERVED_":
            is_in_model = True
            g_vec = google_300.wv[ "reserved" ]
        elif token == "<number>":
            is_in_model = True
            g_vec = google_300.wv[ "number" ]
        elif token == "<percent>":
            is_in_model = True
            g_vec = google_300.wv[ "percent" ]
        elif token == "<money>":
            is_in_model = True
            g_vec = google_300.wv[ "money" ]
        elif token == "<email>":
            is_in_model = True
            g_vec = google_300.wv[ "email" ]
        elif token == "<phone>":
            is_in_model = True
            g_vec = google_300.wv[ "phone" ]
        elif token == "<time>":
            is_in_model = True
            g_vec = google_300.wv[ "time" ]
        elif token == "<date>":
            is_in_model = True
            g_vec = google_300.wv[ "date" ]
        elif not is_in_model:
            max = len ( google_300.wv.vocab.keys ( ) ) - 1
            index = rn.randint ( 0 , max )
            word = google_300.index2word[ index ]
            g_vec = google_300.wv[ word ]

        general_vec = [ ]
        is_in_model = False
        token = token.lower ( )
        if token == "_MENTION_":
            general_vec = generics_100.wv[ "mention" ]
            is_in_model = True
        elif token == "_URL_":
            is_in_model = True
            general_vec = generics_100.wv[ "url" ]
        elif token == "_RESERVED_":
            is_in_model = True
            general_vec = generics_100.wv[ "reserved" ]
        elif token == "<number>":
            is_in_model = True
            general_vec = generics_100.wv[ "number" ]
        elif token == "<percent>":
            is_in_model = True
            general_vec = generics_100.wv[ "percent" ]
        elif token == "<money>":
            is_in_model = True
            general_vec = generics_100.wv[ "money" ]
        elif token == "<email>":
            is_in_model = True
            general_vec = generics_100.wv[ "email" ]
        elif token == "<phone>":
            is_in_model = True
            general_vec = generics_100.wv[ "phone" ]
        elif token == "<time>":
            is_in_model = True
            general_vec = generics_100.wv[ "time" ]
        elif token == "<date>":
            is_in_model = True
            general_vec = generics_100.wv[ "date" ]
        elif not is_in_model:
            max = len ( generics_100.wv.vocab.keys ( ) ) - 1
            index = rn.randint ( 0 , max )
            word = generics_100.index2word[ index ]
            general_vec = generics_100.wv[ word ]

        # VECTOR FOR EACH TOKEN
        tok2D = []
        tok2D = np.concatenate((g_vec,general_vec), axis = 0)
        tok2D = tok2D.tolist()

        tok2D.append ( sent_upper_pec )
        tok2D.append ( sent_repPercent )
        tok2D.append ( numChar )
        tok2D.append ( numTok )

        tok2D.append ( per_numbers )
        tok2D.append ( per_emails )
        tok2D.append ( per_money )
        tok2D.append ( per_phone )
        tok2D.append ( per_time )
        tok2D.append ( per_date )

        tok2D.append ( percEmoticons )
        tok2D.append ( perc_nouns )
        tok2D.append ( perc_verbs )
        tok2D.append ( perc_adj )
        tok2D.append ( perc_pron )
        tok2D.append ( perc_adv )
        tok2D.append ( percWhite )
        tok2D.append ( percEscl )
        tok2D.append ( percQmark )
        tok2D.append ( percStop )
        tok2D.append ( percDic )
        tok2D.append ( pol )
        tok2D.append ( percVpos )
        tok2D.append ( percPos )
        tok2D.append ( percNeutral )
        tok2D.append ( percNeg )
        tok2D.append ( percVNeg )
        tok2D.append ( upperPercent )
        tok2D.append ( repPercent )
        tok2D.append ( polarity )
        tok2D.append ( intensity )
        tok2D.append ( esclamation )
        tok2D.append ( question )
        tok2D.append ( isStop )
        tok2D.append ( isDic )
        tok2D.append ( w_pos )
        tok2D.append ( w_iob )
        tok2D.append ( w_ner )

        if tokIndex < 50:
            matrix3D[indexs][tokIndex] = tok2D
        tokIndex +=1

    indexs = indexs +1

    print ( '------------' , indexs )


joblib.dump(matrix3D, 'matrix3D_google_general_full_dev')


