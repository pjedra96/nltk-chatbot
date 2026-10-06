# Description
This project contains a rule-based chatbot web application for Biodata Ltd (Manchester, UK), written in Python using Flask, NLTK and scikit-learn.

The chatbot does not use AI or a language model. Everything it knows is written by hand in `knowledge.json`, and it answers by matching what the user types against example questions stored in that file. It can:
- make small talk (greetings, jokes, "how are you", goodbyes, etc.)
- share general knowledge facts, or a random fun fact
- answer questions about Biodata Ltd
- address the user by the name they ask for (e.g. "call me Sam")

If the chatbot is not confident about a match, it says it doesn't know rather than guessing.

# System & Package Requirements
- Python 3.12 or newer
- All packages contained in the requirements.txt file

# Installation
1. Clone the repository and open a terminal in its root folder.
2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```
   On macOS / Linux, activate it with `source .venv/bin/activate` instead.
3. Install the packages:
   ```
   pip install -r requirements.txt
   ```
4. Run the chatbot:
   ```
   python bot.py
   ```
   The first run downloads the NLTK data it needs (this only happens once).
5. Open `localhost:5000` in the browser to chat with the bot.

# How it works
1. Each message is tokenised and lemmatised with NLTK (e.g. "bones" becomes "bone").
2. The message is compared with every example question in `knowledge.json` using TF-IDF and cosine similarity (scikit-learn).
3. If the closest example is similar enough, the bot replies with one of that topic's responses, picked at random. Otherwise it gives a fallback reply.

Requests such as "call me ...", "what's my name?" and "forget my name" are handled separately in `bot.py`. Each visitor's name is stored in their browser session and is forgotten when the server restarts.

# Editing the chatbot's knowledge
All of the chatbot's knowledge lives in `knowledge.json`. Each topic (intent) looks like this:
```json
{
  "tag": "company_hours",
  "category": "company",
  "patterns": ["what are your opening hours", "when are you open"],
  "responses": ["Our office hours are Monday to Friday, 9am to 5pm."]
}
```
- **patterns** - different ways a person might ask the question. The more wordings you add, the better the bot recognises the question.
- **responses** - what the bot replies. If there is more than one, one is picked at random.
- Use `{name}` in a response to include the user's name, and `{bot_name}` for the chatbot's name.

Restart `bot.py` after editing the file to load the changes.

Company details shown in `[square brackets]` are placeholders that still need to be filled in.

# Project structure
- `bot.py` - the Flask app and chatbot logic
- `knowledge.json` - the chatbot's topics, questions and answers
- `templates/index.html` - the chat page
- `static/css/style.css` - the chat page styling
- `requirements.txt` - Python packages needed to run the project

# License
This project is licensed under the GNU General Public License v3.0 - see the `LICENSE` file for details.
