import math
from collections import defaultdict, Counter
import re

class SpellingFixerHMM:
    def __init__(self, aspell_file):
        self.word_corrections = {}
        self.word_frequencies = defaultdict(int)
        self.emission_probs = defaultdict(lambda: defaultdict(float))
        self.transition_probs = defaultdict(lambda: defaultdict(float))
        self.start_probs = defaultdict(float)
        self.end_probs = defaultdict(float)
        self.states = set()
        self.observations = set()
        
        # Load aspell data
        self._load_aspell_data(aspell_file)
        
        # Build character-level HMM from word pairs
        self._build_character_hmm()
    
    def _load_aspell_data(self, aspell_file):
        """Load the aspell data and create word mappings."""
        self.word_pairs = []
        
        with open(aspell_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if ':' in line:
                    correct_word, misspellings = line.split(':', 1)
                    correct_word = correct_word.strip().lower()
                    misspellings = [m.strip().lower() for m in misspellings.split()]
                    
                    # Count frequency of correct word
                    self.word_frequencies[correct_word] += 1
                    
                    # Map each misspelling to the correct word
                    for misspelling in misspellings:
                        self.word_corrections[misspelling] = correct_word
                        self.word_pairs.append((correct_word, misspelling))
    
    def _build_character_hmm(self):
        """Build character-level HMM from word pairs using edit distance alignment."""
        emission_counts = defaultdict(lambda: defaultdict(int))
        state_counts = defaultdict(int)
        transition_counts = defaultdict(lambda: defaultdict(int))
        start_counts = defaultdict(int)
        end_counts = defaultdict(int)
        total_words = 0
        
        for correct_word, misspelled_word in self.word_pairs:
            if not correct_word:
                continue
                
            total_words += 1
            
            # Use edit distance to find the best alignment
            aligned_pairs = self._edit_distance_align(correct_word, misspelled_word)
            
            # Extract character mappings from alignment
            for correct_char, typed_char in aligned_pairs:
                if correct_char and typed_char:  # Skip gaps
                    self.states.add(correct_char)
                    self.observations.add(typed_char)
                    emission_counts[correct_char][typed_char] += 1
                    state_counts[correct_char] += 1
            
            # Build transition probabilities from correct words
            word_chars = list(correct_word)
            
            # Start state transitions
            if word_chars:
                start_counts[word_chars[0]] += 1
                self.states.add(word_chars[0])
            
            # State-to-state transitions
            for i in range(len(word_chars) - 1):
                current_char = word_chars[i]
                next_char = word_chars[i + 1]
                transition_counts[current_char][next_char] += 1
                self.states.add(next_char)
            
            # End state transitions
            if word_chars:
                end_counts[word_chars[-1]] += 1
        
        # Calculate emission probabilities with smoothing
        smoothing = 0.01
        for correct_char in self.states:
            total_count = state_counts[correct_char]
            for typed_char in self.observations:
                count = emission_counts[correct_char][typed_char]
                prob = (count + smoothing) / (total_count + smoothing * len(self.observations))
                self.emission_probs[correct_char][typed_char] = prob
        
        # Calculate transition probabilities
        for current_char in self.states:
            total_transitions = sum(transition_counts[current_char].values())
            if total_transitions > 0:
                for next_char in self.states:
                    count = transition_counts[current_char][next_char]
                    self.transition_probs[current_char][next_char] = (count + smoothing) / (total_transitions + smoothing * len(self.states))
            else:
                for next_char in self.states:
                    self.transition_probs[current_char][next_char] = 1.0 / len(self.states)
        
        # Calculate start and end probabilities
        for char in self.states:
            self.start_probs[char] = (start_counts[char] + smoothing) / (total_words + smoothing * len(self.states))
            self.end_probs[char] = (end_counts[char] + smoothing) / (total_words + smoothing * len(self.states))
    
    def _edit_distance_align(self, correct_word, misspelled_word):
        """Use dynamic programming to find the best alignment between two words."""
        m, n = len(correct_word), len(misspelled_word)
        
        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        path = [[None] * (n + 1) for _ in range(m + 1)]
        
        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
            path[i][0] = 'D'  # Deletion
        for j in range(n + 1):
            dp[0][j] = j
            path[0][j] = 'I'  # Insertion
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if correct_word[i-1] == misspelled_word[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                    path[i][j] = 'M'  # Match
                else:
                    # Find minimum cost operation
                    costs = [
                        (dp[i-1][j] + 1, 'D'),      # Deletion
                        (dp[i][j-1] + 1, 'I'),      # Insertion
                        (dp[i-1][j-1] + 1, 'S')     # Substitution
                    ]
                    min_cost, operation = min(costs)
                    dp[i][j] = min_cost
                    path[i][j] = operation
        
        # Backtrack to find alignment
        aligned = []
        i, j = m, n
        
        while i > 0 and j > 0:
            if path[i][j] == 'M':
                aligned.append((correct_word[i-1], misspelled_word[j-1]))
                i -= 1
                j -= 1
            elif path[i][j] == 'D':
                aligned.append((correct_word[i-1], None))
                i -= 1
            elif path[i][j] == 'I':
                aligned.append((None, misspelled_word[j-1]))
                j -= 1
            else:  # 'S' - Substitution
                aligned.append((correct_word[i-1], misspelled_word[j-1]))
                i -= 1
                j -= 1
        
        # Handle remaining characters
        while i > 0:
            aligned.append((correct_word[i-1], None))
            i -= 1
        while j > 0:
            aligned.append((None, misspelled_word[j-1]))
            j -= 1
        
        aligned.reverse()
        return aligned
    
    def viterbi_decode(self, observation_sequence):
        """Decode the observation sequence using the Viterbi algorithm."""
        if not observation_sequence:
            return ""
        
        T = len(observation_sequence)
        N = len(self.states)
        
        if N == 0:
            return observation_sequence
        
        states_list = list(self.states)
        
        # Use log probabilities to avoid underflow
        viterbi = [[float('-inf') for _ in range(N)] for _ in range(T)]
        backpointer = [[0 for _ in range(N)] for _ in range(T)]
        
        # Initialization step
        for s in range(N):
            state = states_list[s]
            obs = observation_sequence[0]
            
            emission_prob = self.emission_probs[state].get(obs, 0.001)
            start_prob = self.start_probs[state]
            
            viterbi[0][s] = math.log(start_prob) + math.log(emission_prob)
            backpointer[0][s] = 0
        
        # Recursion step
        for t in range(1, T):
            for s in range(N):
                current_state = states_list[s]
                obs = observation_sequence[t]
                
                emission_prob = self.emission_probs[current_state].get(obs, 0.001)
                
                max_log_prob = float('-inf')
                best_prev_state = 0
                
                for prev_s in range(N):
                    prev_state = states_list[prev_s]
                    transition_prob = self.transition_probs[prev_state][current_state]
                    
                    log_prob = viterbi[t-1][prev_s] + math.log(transition_prob) + math.log(emission_prob)
                    
                    if log_prob > max_log_prob:
                        max_log_prob = log_prob
                        best_prev_state = prev_s
                
                viterbi[t][s] = max_log_prob
                backpointer[t][s] = best_prev_state
        
        # Termination step
        max_final_log_prob = float('-inf')
        best_final_state = 0
        
        for s in range(N):
            state = states_list[s]
            end_prob = self.end_probs[state]
            log_prob = viterbi[T-1][s] + math.log(end_prob)
            
            if log_prob > max_final_log_prob:
                max_final_log_prob = log_prob
                best_final_state = s
        
        # Backtrack to find the best path
        best_path = []
        current_state = best_final_state
        
        for t in range(T-1, -1, -1):
            best_path.append(states_list[current_state])
            current_state = backpointer[t][current_state]
        
        best_path.reverse()
        return ''.join(best_path)
    
    def correct_text(self, text):
        """Correct spelling errors using both HMM and direct lookup."""
        words = text.split()
        corrected_words = []
        
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if clean_word:
                # First try direct lookup
                if clean_word in self.word_corrections:
                    corrected_word = self.word_corrections[clean_word]
                else:
                    # Use HMM for unknown words
                    corrected_word = self.viterbi_decode(clean_word)
                
                # Preserve original case
                if word[0].isupper():
                    corrected_word = corrected_word.capitalize()
                corrected_words.append(corrected_word)
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words)
    
    def print_statistics(self):
        """Print statistics about the trained model."""
        print(f"Number of states (correct letters): {len(self.states)}")
        print(f"Number of observations (typed letters): {len(self.observations)}")
        print(f"Number of word pairs processed: {len(self.word_pairs)}")
        print(f"Number of direct corrections: {len(self.word_corrections)}")
        
        print("\nTop emission probabilities (P(typed|correct)):")
        for correct_char in sorted(self.states)[:5]:
            top_emissions = sorted(self.emission_probs[correct_char].items(), 
                                 key=lambda x: x[1], reverse=True)[:3]
            print(f"  {correct_char}: {top_emissions}")

def test_spelling_fixer():
    """Test the spelling fixer with various inputs."""
    print("Loading HMM spelling fixer...")
    fixer = SpellingFixerHMM('aspell.txt')
    
    print("\nModel Statistics:")
    fixer.print_statistics()
    
    test_words = [
        "helo", "beleive", "definately", "recieve", 
        "teh", "taht", "accomodate", "seperate"
    ]
    
    print("\n" + "="*60)
    print("TESTING SPECIFIC WORDS")
    print("="*60)
    
    for word in test_words:
        corrected = fixer.correct_text(word)
        print(f"'{word}' -> '{corrected}'")
    
    print("\n" + "="*60)
    print("MULTI-WORD TESTS")
    print("="*60)
    
    test_phrases = [
        "helo wrld",
        "beleive in yorself", 
        "teh quick brown fox",
        "definately recieve"
    ]
    
    for phrase in test_phrases:
        corrected = fixer.correct_text(phrase)
        print(f"'{phrase}' -> '{corrected}'")

if __name__ == "__main__":
    test_spelling_fixer()
