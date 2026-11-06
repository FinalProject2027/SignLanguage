import re

class TextPostProcessor:
    def __init__(self, confidence_threshold: float = 0.70, debounce_window_sec: float = 1.0):
        """Initializes token debounce filters, thresholds, and sentence memory."""
        self.min_confidence = confidence_threshold
        self.debounce_window = debounce_window_sec
        self.token_buffer = []

    def process_sequence(self, decoded_text: str) -> str:
        """
        Cleans the decoded string from the Hugging Face tokenizer.
        Removes consecutive duplicate words and formats capitalization.
        """
        # 1. Clean out residual special tokens
        clean_text = decoded_text.replace("[SOS]", "").replace("[EOS]", "").replace("[PAD]", "").strip()
        
        if not clean_text:
            return ""

        # 2. Split into raw model tokens (preserving underscores so "THANK_YOU" stays one unit)
        tokens = clean_text.split()
        
        # 3. Consecutive Debounce: Remove back-to-back duplicate tokens[cite: 5]
        debounced_tokens = []
        for token in tokens:
            if not debounced_tokens or token.lower() != debounced_tokens[-1].lower():
                debounced_tokens.append(token)
                
        # 4. Re-join, replace underscores with natural spaces, and add punctuation[cite: 5]
        smoothed_sentence = " ".join(debounced_tokens)
        smoothed_sentence = smoothed_sentence.replace("_", " ")
        
        if smoothed_sentence:
            smoothed_sentence = smoothed_sentence.capitalize() + "."
            
        return smoothed_sentence