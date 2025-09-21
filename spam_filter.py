import numpy as np
import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from difflib import SequenceMatcher

class SpamFilterTFIDF:
    def __init__(self):
        """
        Initialize the spam filter with spelling mistake detection
        """
        self.vocabulary = []
        self.documents = []
        self.labels = []
        self.tf_idf_matrix = None
        self.classifier = MultinomialNB()
        self.spelling_tricks = {}  # Will store common spam spelling tricks
        
    def setup_spelling_tricks(self):
        """
        Setup common spelling tricks used by spammers
        
        What this does:
        - Creates a dictionary of how spammers disguise words
        - Maps misspelled versions back to correct vocabulary words
        - Helps catch tricks like "fr33" -> "free", "m0ney" -> "money"
        """
        print("=== Setting up Spelling Trick Detection ===")
        
        # Common letter-to-number substitutions used by spammers
        self.letter_to_number = {
            'a': ['@', '4'], 'e': ['3'], 'i': ['1', '!'], 'o': ['0'], 
            's': ['5', '$'], 't': ['7'], 'l': ['1'], 'g': ['9']
        }
        
        # Generate spelling variations for each vocabulary word
        for word in self.vocabulary:
            variations = self.generate_spelling_variations(word)
            for variation in variations:
                self.spelling_tricks[variation] = word
        
        print(f"Generated {len(self.spelling_tricks)} spelling variations")
        print("Examples of detected spelling tricks:")
        count = 0
        for trick, original in self.spelling_tricks.items():
            if trick != original and count < 10:  # Show first 10 examples
                print(f"  {trick} -> {original}")
                count += 1
        print()
    
    def generate_spelling_variations(self, word):
        """
        Generate common spelling variations for a word
        
        What this does:
        - Takes a word like "free"
        - Creates variations like "fr33", "fr3e", "f@ee", etc.
        - Returns all possible spam-style misspellings
        """
        variations = [word]  # Start with original word
        
        # Generate number substitutions
        for i, char in enumerate(word):
            if char.lower() in self.letter_to_number:
                for replacement in self.letter_to_number[char.lower()]:
                    new_word = word[:i] + replacement + word[i+1:]
                    variations.append(new_word)
                    
                    # Also try multiple substitutions (like "fr33" for "free")
                    if len(word) > 3:
                        for j, char2 in enumerate(word):
                            if j != i and char2.lower() in self.letter_to_number:
                                for replacement2 in self.letter_to_number[char2.lower()]:
                                    double_sub = new_word[:j] + replacement2 + new_word[j+1:]
                                    variations.append(double_sub)
        
        # Add some common spam patterns
        variations.extend([
            word.upper(),  # ALL CAPS
            word + '!',    # Extra exclamation
            word + '!!',   # Multiple exclamations
            word.replace('e', '').replace('a', ''),  # Missing vowels
        ])
        
        return list(set(variations))  # Remove duplicates
    
    def similarity_score(self, word1, word2):
        """
        Calculate similarity between two words (0 to 1, where 1 is identical)
        
        This helps catch misspellings that our predefined tricks might miss
        """
        return SequenceMatcher(None, word1.lower(), word2.lower()).ratio()
    
    def find_closest_vocab_word(self, word):
        """
        Find the closest vocabulary word to a potentially misspelled word
        
        What this does:
        - Takes a word that might be misspelled
        - Compares it to all vocabulary words
        - Returns the closest match ONLY if it's clearly a spam trick
        - IMPROVED: Avoids false positives from innocent human typos
        """
        best_match = None
        best_score = 0.0
        
        # Different thresholds for different types of matches
        spam_trick_threshold = 0.6   # Lower threshold for known spam patterns
        typo_threshold = 0.85        # Higher threshold for potential innocent typos
        
        for vocab_word in self.vocabulary:
            score = self.similarity_score(word, vocab_word)
            
            # Check if this looks like a deliberate spam trick (has numbers/symbols)
            has_numbers_or_symbols = bool(re.search(r'[0-9@!$]', word))
            
            if has_numbers_or_symbols:
                # More lenient for obvious spam tricks like "fr33", "m0ney"
                if score > best_score and score >= spam_trick_threshold:
                    best_score = score
                    best_match = vocab_word
            else:
                # Stricter for regular words to avoid false positives
                # Only match if very similar (likely same word with small typo)
                if score > best_score and score >= typo_threshold:
                    best_score = score
                    best_match = vocab_word
        
        return best_match, best_score
        
    def load_vocabulary(self, vocab_text):
        """
        Load and clean the vocabulary from the dictionary
        
        What this does:
        - Takes the spam dictionary text
        - Cleans it up (removes spaces, converts to lowercase)
        - Creates a list of important spam words to look for
        - Sets up spelling trick detection
        """
        print("=== STEP 1: Loading Vocabulary ===")
        
        # Clean the vocabulary text and split into words
        vocab_words = re.findall(r'[a-zA-Z]+', vocab_text.lower())
        # Remove duplicates and keep unique words
        self.vocabulary = list(set(vocab_words))
        
        print(f"Loaded {len(self.vocabulary)} unique vocabulary words:")
        print(self.vocabulary)
        print()
        
        # NEW: Setup spelling trick detection
        self.setup_spelling_tricks()
        
    def preprocess_text(self, text):
        """
        Clean and prepare text for analysis with spelling mistake detection
        
        What this does:
        - Converts text to lowercase
        - Removes special characters and numbers (but keeps them for spell check first)
        - Detects spelling tricks and maps them back to correct words
        - IMPROVED: Distinguishes between spam tricks and innocent human typos
        - Returns words found in vocabulary (including corrected spellings)
        """
        # First, extract words including numbers and symbols for spell checking
        raw_words = re.findall(r'[a-zA-Z0-9@!$]+', text.lower())
        
        corrected_words = []
        spelling_corrections = []
        
        for word in raw_words:
            # Clean the word (remove numbers/symbols for final processing)
            clean_word = re.sub(r'[^a-zA-Z]', '', word)
            
            # Skip very short words
            if len(clean_word) < 2:
                continue
            
            # Check if it's directly in vocabulary
            if clean_word in self.vocabulary:
                corrected_words.append(clean_word)
            
            # Check if it's a known spelling trick
            elif word in self.spelling_tricks:
                original_word = self.spelling_tricks[word]
                corrected_words.append(original_word)
                spelling_corrections.append(f"{word} -> {original_word} (spam trick)")
            
            # Check for similar words (fuzzy matching)
            else:
                closest_word, similarity = self.find_closest_vocab_word(word)
                if closest_word and similarity > 0.6:
                    # Determine if this looks like a spam trick or innocent typo
                    has_spam_patterns = bool(re.search(r'[0-9@!$]', word))
                    
                    if has_spam_patterns:
                        # Likely spam trick - include it
                        corrected_words.append(closest_word)
                        spelling_corrections.append(f"{word} -> {closest_word} (spam trick, similarity: {similarity:.2f})")
                    elif similarity >= 0.85:
                        # Very high similarity - likely innocent typo of spam word
                        corrected_words.append(closest_word)
                        spelling_corrections.append(f"{word} -> {closest_word} (possible typo, similarity: {similarity:.2f})")
                    # If similarity is between 0.6-0.85 for normal words, ignore it
                    # This avoids false positives like "food" -> "free"
        
        # Print spelling corrections found (for debugging/transparency)
        if spelling_corrections:
            print(f"  Spelling corrections detected: {spelling_corrections}")
        
        return corrected_words
    
    def load_documents(self, documents_data):
        """
        Load and preprocess all training documents
        
        What this does:
        - Takes all 6 spam documents
        - Cleans each document
        - Stores them for TF-IDF calculation
        """
        print("=== STEP 2: Loading and Preprocessing Documents ===")
        
        for i, doc in enumerate(documents_data, 1):
            # Clean the document text
            processed_doc = self.preprocess_text(doc)
            self.documents.append(processed_doc)
            self.labels.append('spam')  # All our training documents are spam
            
            print(f"Document {i}:")
            print(f"  Processed words: {processed_doc}")
            print(f"  Word count: {len(processed_doc)}")
            print()
    
    def calculate_tf(self, document):
        """
        Calculate Term Frequency (TF) for a document
        
        TF = (Number of times a word appears in document) / (Total words in document)
        
        Example: If "free" appears 2 times in a document with 10 words total,
        TF for "free" = 2/10 = 0.2
        """
        doc_length = len(document)
        tf_dict = {}
        
        for word in self.vocabulary:
            word_count = document.count(word)
            tf_dict[word] = word_count / doc_length if doc_length > 0 else 0
            
        return tf_dict
    
    def calculate_idf(self):
        """
        Calculate Inverse Document Frequency (IDF) for all vocabulary words
        
        IDF = log(Total number of documents / Number of documents containing the word)
        
        This gives higher weights to rare words and lower weights to common words.
        Example: If "free" appears in all 6 documents, IDF is low.
        If "pension" appears in only 1 document, IDF is high.
        """
        total_documents = len(self.documents)
        idf_dict = {}
        
        for word in self.vocabulary:
            # Count how many documents contain this word
            containing_docs = sum(1 for doc in self.documents if word in doc)
            # Calculate IDF (add 1 to avoid division by zero)
            idf_dict[word] = np.log(total_documents / (containing_docs + 1))
            
        return idf_dict
    
    def calculate_tfidf_matrix(self):
        """
        Calculate TF-IDF matrix for all documents
        
        TF-IDF = TF * IDF
        
        This combines:
        - How often a word appears in a document (TF)
        - How rare the word is across all documents (IDF)
        """
        print("=== STEP 3: Calculating TF-IDF Scores ===")
        
        # Calculate IDF for all words
        idf_dict = self.calculate_idf()
        
        # Create matrix to store TF-IDF scores
        tfidf_matrix = []
        
        print("IDF Scores (higher = more unique/important):")
        for word, idf in idf_dict.items():
            print(f"  {word}: {idf:.3f}")
        print()
        
        # Calculate TF-IDF for each document
        for doc_idx, document in enumerate(self.documents):
            print(f"Document {doc_idx + 1} TF-IDF scores:")
            
            # Calculate TF for this document
            tf_dict = self.calculate_tf(document)
            
            # Calculate TF-IDF for each word
            tfidf_vector = []
            for word in self.vocabulary:
                tf_score = tf_dict[word]
                idf_score = idf_dict[word]
                tfidf_score = tf_score * idf_score
                tfidf_vector.append(tfidf_score)
                
                if tfidf_score > 0:  # Only show words that appear in the document
                    print(f"  {word}: TF={tf_score:.3f} × IDF={idf_score:.3f} = {tfidf_score:.3f}")
            
            tfidf_matrix.append(tfidf_vector)
            print()
        
        self.tf_idf_matrix = np.array(tfidf_matrix)
        return self.tf_idf_matrix
    
    def train_classifier(self):
        """
        Train a machine learning classifier using TF-IDF features
        
        What this does:
        - Uses the TF-IDF scores as features
        - Trains a Naive Bayes classifier (good for text classification)
        - This will help classify new messages as spam or not spam
        """
        print("=== STEP 4: Training Classifier ===")
        
        if self.tf_idf_matrix is None:
            self.calculate_tfidf_matrix()
        
        # Train the classifier
        self.classifier.fit(self.tf_idf_matrix, self.labels)
        print("Classifier trained successfully!")
        print()
    
    def predict_spam(self, new_text):
        """
        Predict if a new message is spam or not
        
        What this does:
        - Cleans the new message
        - Calculates its TF-IDF scores
        - Uses the trained classifier to make a prediction
        - IMPROVED: Handles cases with no vocabulary words
        """
        # Preprocess the new text
        processed_text = self.preprocess_text(new_text)
        
        # IMPROVEMENT: Handle messages with no vocabulary words
        if len(processed_text) == 0:
            return "not_spam", [0.1, 0.9], processed_text, {}
        
        # Calculate TF for the new document
        tf_dict = self.calculate_tf(processed_text)
        
        # Calculate IDF (reuse from training)
        idf_dict = {}
        total_documents = len(self.documents)
        for word in self.vocabulary:
            containing_docs = sum(1 for doc in self.documents if word in doc)
            idf_dict[word] = np.log(total_documents / (containing_docs + 1))
        
        # Calculate TF-IDF vector for new text
        tfidf_vector = []
        spam_word_count = 0
        for word in self.vocabulary:
            tf_score = tf_dict[word]
            idf_score = idf_dict[word]
            tfidf_score = tf_score * idf_score
            tfidf_vector.append(tfidf_score)
            if tfidf_score > 0:
                spam_word_count += 1
        
        # IMPROVEMENT: Apply spam word threshold
        tfidf_vector = np.array(tfidf_vector).reshape(1, -1)
        
        # If very few spam words detected, lean towards not_spam
        if spam_word_count <= 1:
            # Still use classifier but adjust confidence
            prediction = self.classifier.predict(tfidf_vector)[0]
            raw_probability = self.classifier.predict_proba(tfidf_vector)[0]
            
            # Reduce confidence in spam prediction when few spam words present
            adjusted_probability = [raw_probability[0] * 0.3, raw_probability[1] * 0.3 + 0.7]
            final_prediction = "spam" if adjusted_probability[1] > 0.5 else "not_spam"
            
            return final_prediction, adjusted_probability, processed_text, dict(zip(self.vocabulary, tfidf_vector[0]))
        else:
            # Normal prediction for messages with multiple spam words
            prediction = self.classifier.predict(tfidf_vector)[0]
            probability = self.classifier.predict_proba(tfidf_vector)[0]
            return prediction, probability, processed_text, dict(zip(self.vocabulary, tfidf_vector[0]))
    
    def display_feature_importance(self):
        """
        Show which words are most important for spam detection
        """
        print("=== STEP 5: Feature Importance Analysis ===")
        
        # Calculate average TF-IDF scores for each word
        avg_tfidf = np.mean(self.tf_idf_matrix, axis=0)
        
        # Create a DataFrame for better visualization
        feature_importance = pd.DataFrame({
            'Word': self.vocabulary,
            'Average_TFIDF': avg_tfidf
        }).sort_values('Average_TFIDF', ascending=False)
        
        print("Most important words for spam detection:")
        print(feature_importance.head(10))
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.bar(feature_importance['Word'][:10], feature_importance['Average_TFIDF'][:10])
        plt.title('Top 10 Most Important Words for Spam Detection')
        plt.xlabel('Words')
        plt.ylabel('Average TF-IDF Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        return feature_importance

# ============================================================================
# MAIN EXECUTION - Let's build and test our spam filter!
# ============================================================================

def main():
    print("🚀 Building TF-IDF Spam Filter")
    print("=" * 50)
    
    # Initialize the spam filter
    spam_filter = SpamFilterTFIDF()
    
    # Vocabulary from the dictionary
    vocab_text = """Free Click here visit open attachment call this number money Out extra offer available Pension Opportunity Chance Investment Pension"""
    
    # Training documents (the 6 spam messages)
    training_docs = [
        "Free-Coupons for next movie. The above links will take you straight to our partner's site. For more information or to see other offers available, you can also visit the Groupon on the Working Advantage website.",
        
        "Free-Coupons for next movie. The above links will take you straight to our partner's site. For more information or to see other offers available, you can also visit the Groupon on the Working Advantage website.",
        
        "Our records indicate your Pension is under performing to see higher growth and up to 25% cash release reply PENSION for a free review. To opt out reply STOP",
        
        "Enter to win $25,000 and get a Free Hotel Night! Just click here for a $1 trial membership in NetMarket, the Internet's premier discount shopping site: Fast Company EZVenture gives you FREE business articles, PLUS, you could win YOUR CHOICE of a BMW Z3 convertible, $100,000, shares of Microsoft stock, or a home office computer. Go there and get your chances to win now. A crazy-funny-cool trivia book with a $10,000 prize? PLUS chocolate, nail polish, cats, barnyard animals, and more?",
        
        "Dear recipient, Avangar Technologies announces the beginning of a new unprecedented global employment campaign. Due to company's exploding growth Avangar is expanding business to the European region. During last employment campaign over 1500 people worldwide took part in Avangar's business and more than half of them are currently employed by the company. And now we are offering you one more opportunity to earn extra money working with Avangar Technologies. We are looking for honest, responsible, hard-working people that can dedicate 2-4 hours of their time per day and earn extra £300-500 weekly. All offered positions are currently part-time and give you a chance to work mainly from home.",
        
        "I know that's an incredible statement, but bear with me while I explain. You have already deleted mail from dozens of Get Rich Quick schemes, chain letter offers, and LOTS of other absurd scams that promise to make you rich overnight with no investment and no work. My offer isn't one of those. What I'm offering is a straightforward computer-based service that you can run full-or part-time like a regular business. This service runs automatically while you sleep, vacation, or work a regular job. It provides a valuable new service for businesses in your area. I'm offering a high-tech, low-maintenance, work-from-anywhere business that can bring in a nice comfortable additional income for your family. I did it for eight years. Since I started inviting others to join me, I've helped over 4000 do the same."
    ]
    
    # Step 1: Load vocabulary
    spam_filter.load_vocabulary(vocab_text)
    
    # Step 2: Load and preprocess documents
    spam_filter.load_documents(training_docs)
    
    # Step 3: Calculate TF-IDF matrix
    spam_filter.calculate_tfidf_matrix()
    
    # Step 4: Train classifier
    spam_filter.train_classifier()
    
    # Step 5: Display feature importance
    feature_importance = spam_filter.display_feature_importance()
    
    # Step 6: Test with new messages
    print("\n" + "=" * 50)
    print("🧪 TESTING THE SPAM FILTER")
    print("=" * 50)
    
    test_messages = [
        "Free money! Click here to get rich quick!",  # Should be SPAM (has: free, money, click, here)
        "Hi mom, how are you doing today?",           # Should be NOT SPAM (no spam words)
        "Get extra cash with this amazing opportunity!", # Should be SPAM (has: extra, opportunity)
        "Meeting at 3pm in conference room",          # Should be NOT SPAM (no spam words)
        "Your pension needs review - call this number now!", # Should be SPAM (has: pension, call, this, number)
        "Hi my name is pranjal!",                     # Should be NOT SPAM (no spam words)
        "Free offer available click here",            # Should be SPAM (has: free, offer, available, click, here)
        "What time is the meeting?",                  # Should be NOT SPAM (no spam words)
        
        # NEW: Test spelling mistake detection vs innocent typos
        "Fr33 m0ney! C1ick h3re n0w!",              # Should be SPAM (free, money, click, here with numbers)
        "F@EE 0FF3R 4V41L4BLE!",                    # Should be SPAM (free, offer, available with symbols)
        "Get 3xtra ca$h with thi5 0pp0rtun1ty",     # Should be SPAM (extra, cash, this, opportunity)
        "Your pen5ion investment opportunity",       # Should be SPAM (pension, investment, opportunity)
        "C4LL th15 numb3r for mon3y",               # Should be SPAM (call, this, number, money)
        "Regular meeting at office",                 # Should be NOT SPAM (no spam words, no tricks)
        
        # NEW: Test innocent human typos vs spam
        "I want to earm some money honestly",        # Should be NOT SPAM (earm≠earn, no spam context)
        "Please claick on the legitimate link",      # Should be NOT SPAM (innocent typo, no spam words)
        "Freee pizza at the office party!",         # Should be NOT SPAM (innocent typo, not spam context)
        "Investment opportunaty available",          # Should be SPAM (opportunaty→opportunity, spam word)
        "Freee money click heere now!",             # Should be SPAM (multiple spam words even with typos)
    ]
    
    for i, message in enumerate(test_messages, 1):
        print(f"\nTest Message {i}: '{message}'")
        prediction, probability, processed_words, tfidf_scores = spam_filter.predict_spam(message)
        
        print(f"Prediction: {prediction.upper()}")
        print(f"Confidence: {max(probability):.1%}")
        print(f"Processed words found: {processed_words}")
        
        # Show which words contributed to the decision
        important_words = {word: score for word, score in tfidf_scores.items() if score > 0}
        if important_words:
            print("Important words detected:")
            for word, score in sorted(important_words.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {word}: {score:.3f}")
        print("-" * 30)

if __name__ == "__main__":
    main()