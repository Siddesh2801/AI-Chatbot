# imports
from flask import Flask, render_template, request
import Chatbot
app = Flask(__name__)
# create chatbot


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get")
# function for the bot response
def get_bot_response():
    userText = request.args.get('msg')
    bot_response = Chatbot.chat(userText)
    return str(bot_response)


if __name__ == "__main__":
    app.run(debug=True)
