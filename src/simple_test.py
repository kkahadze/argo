#!/usr/bin/env python3
import sys
import os
from extract_definition import extract_definition

# Dictionary of test cases: lemma -> expected Georgian definition
TEST_CASES = {
    "skua": "შვილი",
    "skirapil-i": "მშრალი",
    "jgir-i": "კარგი",
    "okotome": "საქათმე",
    "jima": "ძმა",
    "se'borua": "ნაწილ-ნაწილ",
    "supur-i": "არაფრის მქონე",
    "ch'erch'e": "ჭრელი",
    "՚val-i": "ყველი",          # Cheese
    "՚valam-i": "ყველიანი",     # With cheese
    "q'ilo": "ვეფხვი",          # Leopard
    "ch'el-i": "შეკერილი",      # Sewn
}

def run_test(lemma, expected_definition, dictionary_path="../kajaia.txt"):
    """Run a single test case."""
    # Find the dictionary entry
    entry_text = None
    with open(dictionary_path, 'r', encoding='utf-8') as file:
        current_entry = []
        recording = False
        
        for line in file:
            if line.strip().startswith('Lemma:'):
                # If we were recording an entry, return it as we've reached the next lemma
                if recording and current_entry:
                    entry_text = '\n'.join(current_entry)
                    break
                
                # Check if this is the lemma we're looking for
                current_lemma = line.replace('Lemma:', '').strip()
                if lemma.lower() == current_lemma.lower() or lemma.lower() == current_lemma.lower().replace('-', ''):
                    recording = True
                    current_entry = [line.strip()]
                else:
                    recording = False
                    current_entry = []
            
            # Continue recording the current entry
            elif recording:
                current_entry.append(line.strip())
        
        # Check the last entry in the file
        if recording and current_entry:
            entry_text = '\n'.join(current_entry)
    
    if not entry_text:
        print(f"❌ FAIL | {lemma} → ENTRY NOT FOUND")
        return False
    
    # Extract the definition
    _, actual_definition, _ = extract_definition(entry_text)
    
    result = actual_definition == expected_definition
    status = "✅ PASS" if result else "❌ FAIL"
    
    print(f"{status} | {lemma} → {actual_definition}")
    
    if not result:
        print(f"       Expected: {expected_definition}")
        if lemma in ["՚val-i", "q'ilo"]:
            print(f"       Entry text: {entry_text}")
        
    return result

def main():
    # Check if dictionary file exists
    dictionary_path = "../kajaia.txt"
    if not os.path.exists(dictionary_path):
        print(f"Error: Dictionary file '{dictionary_path}' not found.")
        sys.exit(1)
        
    print(f"Testing {len(TEST_CASES)} lemma-definition pairs...\n")
    
    passed = 0
    for lemma, expected in TEST_CASES.items():
        if run_test(lemma, expected, dictionary_path):
            passed += 1
    
    print(f"\nResults: {passed}/{len(TEST_CASES)} tests passed")
    
    # Return success if all tests pass
    return passed == len(TEST_CASES)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 