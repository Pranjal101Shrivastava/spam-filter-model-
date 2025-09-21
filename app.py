from flask import Flask, render_template, request, jsonify
import sys
import os
from spam_filter import SpamFilterTFIDF

# Initialize Flask app
app = Flask(__name__)

# Initialize the spam filter (global variable so it's loaded once)
spam_filter = None

def load_spam_filter():
    """Load and train the spam filter once when the app starts"""
    global spam_filter
    
    if spam_filter is None:
        print("Loading spam filter...")
        spam_filter = SpamFilterTFIDF()
        
        # Vocabulary from the dictionary
        vocab_text = """Free Click here visit open attachment call this number money Out extra offer available Pension Opportunity Chance Investment Pension"""
        
        # Training documents (the 6 spam messages)
        training_docs = [
            "Free-Coupons for next movie. The above links will take you straight to our partner's site. For more information or to see other offers available, you can also visit the Groupon on the Working Advantage website.",
            "Free-Coupons for next movie. The above links will take you straight to our partner's site. For more information or to see other offers available, you can also visit the Groupon on the Working Advantage website.",
            "Our records indicate your Pension is under performing to see higher growth and up to 25% cash release reply PENSION for a free review. To opt out reply STOP",
            "Enter to win $25,000 and get a Free Hotel Night! Just click here for a $1 trial membership in NetMarket, the Internet's premier discount shopping site: Fast Company EZVenture gives you FREE business articles, PLUS, you could win YOUR CHOICE of a BMW Z3 convertible, $100,000, shares of Microsoft stock, or a home office computer. Go there and get your chances to win now.",
            "Dear recipient, Avangar Technologies announces the beginning of a new unprecedented global employment campaign. Due to company's exploding growth Avangar is expanding business to the European region. During last employment campaign over 1500 people worldwide took part in Avangar's business and more than half of them are currently employed by the company. And now we are offering you one more opportunity to earn extra money working with Avangar Technologies.",
            "I know that's an incredible statement, but bear with me while I explain. You have already deleted mail from dozens of Get Rich Quick schemes, chain letter offers, and LOTS of other absurd scams that promise to make you rich overnight with no investment and no work. My offer isn't one of those. What I'm offering is a straightforward computer-based service that you can run full-or part-time like a regular business."
        ]
        
        # Load and train the model
        spam_filter.load_vocabulary(vocab_text)
        spam_filter.load_documents(training_docs)
        spam_filter.calculate_tfidf_matrix()
        spam_filter.train_classifier()
        
        print("Spam filter loaded and trained successfully!")

@app.route('/')
def index():
    """Main page with the spam detection interface"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to predict if a message is spam"""
    try:
        # Get the message from the request
        data = request.json
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'Please enter a message'}), 400
        
        # Make prediction
        prediction, probability, processed_words, tfidf_scores = spam_filter.predict_spam(message)
        
        # Get important words that contributed to the decision
        important_words = {word: score for word, score in tfidf_scores.items() if score > 0}
        sorted_words = sorted(important_words.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Prepare response
        response = {
            'prediction': prediction.upper(),
            'confidence': round(max(probability) * 100, 1),
            'spam_probability': round(probability[1] * 100, 1) if len(probability) > 1 else 0,
            'processed_words': processed_words,
            'important_words': [{'word': word, 'score': round(score, 3)} for word, score in sorted_words],
            'message_length': len(message.split()),
            'spam_words_found': len(processed_words)
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    # Load the spam filter when the app starts
    load_spam_filter()
    
    # Run the Flask app
    print("\n🚀 Starting Spam Filter Web Interface...")
    print("📧 Visit http://localhost:5000 to test your spam filter!")
    print("🔧 Press Ctrl+C to stop the server")
    
    app.run(debug=True, host='0.0.0.0', port=5000)