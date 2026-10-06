#import necessary libraries
from flask import Flask, render_template, request, session
import json
import os
import random
import re
import string # to process standard python strings
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.stem import WordNetLemmatizer
for package in ('punkt', 'punkt_tab', 'wordnet'):
    nltk.download(package, quiet=True) # only downloads the first time

app = Flask(__name__, static_url_path='', static_folder='static', template_folder='templates')
app.secret_key = os.environ.get('SECRET_KEY') or os.urandom(24) # needed to remember each visitor's name

# How similar a message must be to a known pattern (0 to 1) before the bot trusts the match
MATCH_THRESHOLD = 0.4


#Reading in the knowledge file
with open('knowledge.json', 'r', encoding='utf8') as fin:
    knowledge = json.load(fin)

bot_name = knowledge['bot_name']
name_handling = knowledge['name_handling']


# Preprocessing
lemmer = WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# Learn every pattern from the knowledge file
patterns = []
pattern_intents = []
for intent in knowledge['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        pattern_intents.append(intent)

# No stop words: short phrases like "how are you" are made entirely of them
TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, token_pattern=None, ngram_range=(1, 2))
pattern_vectors = TfidfVec.fit_transform(patterns)

def content_words(text):
    """The meaningful words in a text, e.g. "what is the largest planet" -> {"largest", "planet"}"""
    return set(LemNormalize(text)) - ENGLISH_STOP_WORDS

# Every meaningful word used in each intent's patterns
intent_words = {intent['tag']: set().union(*(content_words(p) for p in intent['patterns']))
                for intent in knowledge['intents']}

def best_intent(user_message):
    """Return the intent whose pattern is most similar to the message, or None if nothing is close enough"""
    vals = cosine_similarity(TfidfVec.transform([user_message]), pattern_vectors)[0]
    idx = vals.argmax()
    if vals[idx] < MATCH_THRESHOLD:
        return None
    intent = pattern_intents[idx]
    # Most of the message's meaningful words must belong to the matched intent. This stops
    # "what is quantum physics" matching "what is biodata", or "capital of germany" matching "capital of france"
    message_words = content_words(user_message)
    if message_words and len(message_words & intent_words[intent['tag']]) <= len(message_words) / 2:
        return None
    return intent


# Addressing the user by the name they ask for
FORGET_NAME = re.compile(r"\b(forget my name|stop calling me|don'?t call me)\b", re.I)
ASK_NAME = re.compile(r"\b(what'?s my name|what is my name|who am i|what do you call me|do you know my name)\b", re.I)
SET_NAME = re.compile(
    r"\b(?:call me|address me as|refer to me as|my name is|my name'?s|i'?m called|i am called|i go by|you can call me)\s+(.+)",
    re.I)

def clean_name(raw):
    """Tidy up the name the user gave, or return None if it doesn't look like a name"""
    name = re.split(r"[!?,;]| please\b| thanks?\b| thank you\b", raw, flags=re.I)[0].strip(" .'\"")
    if not name or len(name) > 40 or len(name.split()) > 4:
        return None
    if not re.fullmatch(r"[^\W\d_]+(?:[ .'-]+[^\W\d_]+)*\.?", name): # letters, plus spaces, dots, hyphens and apostrophes
        return None
    if name.islower():
        name = name.title()
    return name

def fill(text):
    """Insert the bot's and the user's names into a response"""
    text = text.replace('{bot_name}', bot_name)
    name = session.get('name')
    if name:
        return text.replace('{name}', name)
    return re.sub(r",? ?\{name\}", '', text) # "Hi {name}!" becomes "Hi!" when no name is known

def handle_name(user_message):
    """Deal with name requests; returns a reply, or None if the message isn't about the user's name"""
    if FORGET_NAME.search(user_message):
        session.pop('name', None)
        return random.choice(name_handling['forget_responses'])
    if ASK_NAME.search(user_message):
        key = 'tell_responses' if session.get('name') else 'unknown_responses'
        return random.choice(name_handling[key])
    match = SET_NAME.search(user_message)
    if match:
        name = clean_name(match.group(1))
        if name:
            session['name'] = name
            return random.choice(name_handling['set_responses'])
        return random.choice(name_handling['invalid_responses'])
    return None


# Generating response
def response(user_message):
    reply = handle_name(user_message)
    if reply is None:
        intent = best_intent(user_message)
        if intent is None:
            reply = random.choice(knowledge['fallback_responses'])
        else:
            reply = random.choice(intent['responses'])
    return fill(reply)


@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/message')
def get_bot_response():
    user_message = (request.args.get('message') or '').strip()
    if not user_message:
        return ''
    return response(user_message)

if __name__ == '__main__':
    app.run()
