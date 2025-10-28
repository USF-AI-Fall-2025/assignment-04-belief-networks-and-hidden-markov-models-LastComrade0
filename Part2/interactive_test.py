from hidden_markov import SpellingFixerHMM

def interactive_test():
    """Interactive testing of the spelling fixer."""
    print("Loading HMM spelling fixer...")
    fixer = SpellingFixerHMM('aspell.txt')
    
    print("\n" + "="*60)
    print("INTERACTIVE SPELLING FIXER TEST")
    print("="*60)
    print("Enter words or phrases to correct (or 'quit' to exit):")
    
    while True:
        try:
            user_input = input("\n> ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if user_input:
                corrected = fixer.correct_text(user_input)
                print(f"Corrected: {corrected}")
            else:
                print("Please enter some text to correct.")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    interactive_test()
