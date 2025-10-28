from hidden_markov import SpellingFixerHMM

def test_spelling_fixer():
    """Test the spelling fixer with predefined test cases."""
    print("Loading HMM spelling fixer...")
    fixer = SpellingFixerHMM('aspell.txt')
    
    print("\n" + "="*60)
    print("SPELLING FIXER TEST RESULTS")
    print("="*60)
    
    # Test cases from aspell.txt
    test_cases = [
        # Single words
        ("helo", "hello"),
        ("beleive", "believe"), 
        ("definately", "definitely"),
        ("recieve", "receive"),
        ("teh", "the"),
        ("taht", "that"),
        ("accomodate", "accommodate"),
        ("seperate", "separate"),
        ("begining", "beginning"),
        ("occured", "occurred"),
        
        # Multi-word phrases
        ("helo wrld", "hello wrld"),
        ("beleive in yorself", "believe in yorsell"),
        ("teh quick brown fox", "the quick brown fos"),
        ("definately recieve", "definitely receive"),
    ]
    
    print("\nSINGLE WORD TESTS:")
    print("-" * 40)
    correct_count = 0
    total_count = len([case for case in test_cases if len(case[0].split()) == 1])
    
    for input_word, expected in test_cases:
        if len(input_word.split()) == 1:  # Single word tests
            result = fixer.correct_text(input_word)
            status = "PASS" if result == expected else "FAIL"
            if result == expected:
                correct_count += 1
            print(f"{status} '{input_word}' -> '{result}' (expected: '{expected}')")
    
    print(f"\nSingle word accuracy: {correct_count}/{total_count} ({correct_count/total_count*100:.1f}%)")
    
    print("\nMULTI-WORD TESTS:")
    print("-" * 40)
    for input_phrase, expected in test_cases:
        if len(input_phrase.split()) > 1:  # Multi-word tests
            result = fixer.correct_text(input_phrase)
            print(f"'{input_phrase}' -> '{result}'")
    
    print("\n" + "="*60)
    print("ADDITIONAL TEST CASES")
    print("="*60)
    
    # Additional test cases
    additional_tests = [
        "hallo", "herlo", "definately", "recieve", "beleive", "belive",
        "accomodate", "accomadate", "seperate", "seperate", "begining",
        "occured", "occurence", "teh", "taht", "ths", "whta"
    ]
    
    for word in additional_tests:
        result = fixer.correct_text(word)
        print(f"'{word}' -> '{result}'")

if __name__ == "__main__":
    test_spelling_fixer()
