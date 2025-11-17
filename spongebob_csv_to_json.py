#!/usr/bin/env python3
import csv
import json
import sys
import html
import re
from pathlib import Path

INPUT = Path('qr_pairs.csv')
OUTPUT = Path('spongebob_dataset.json')

def main():
    if not INPUT.exists():
        print(f"Input CSV not found: {INPUT}", file=sys.stderr)
        sys.exit(1)

    # Allow very large CSV fields
    try:
        csv.field_size_limit(sys.maxsize)
    except Exception:
        pass

    written = 0
    with INPUT.open('r', encoding='utf-8', newline='') as f_in, OUTPUT.open('w', encoding='utf-8') as f_out:
        reader = csv.DictReader(f_in)
        # Validate expected headers
        expected_left = 'Non-SpongeBob Dialogue'
        expected_right = 'SpongeBob Response'
        if expected_left not in reader.fieldnames or expected_right not in reader.fieldnames:
            print(f"Unexpected headers. Found: {reader.fieldnames}", file=sys.stderr)
            sys.exit(2)

        f_out.write('[\n')
        first = True
        for row in reader:
            left = row.get(expected_left, '')
            right = row.get(expected_right, '')
            if not left or not right:
                continue

            # Normalize/clean
            left = html.unescape(left).strip()
            right = html.unescape(right).strip()

            # Remove speaker tags like "Mrs. Puff:", "Mr. Krabs:", "Police Officer #2:",
            # possibly repeated within the string. Keep surrounding spacing/punctuation.
            # Pattern: start or whitespace/([([), then Capitalized phrase ending with colon.
            speaker_pat = re.compile(r'(\A|[\s\[(])([A-Z][\w .\-#\'\(\)]{0,60}?):\s')

            def strip_tags(text: str) -> str:
                prev = None
                # Iterate until no more substitutions
                while prev != text:
                    prev = text
                    text = speaker_pat.sub(lambda m: m.group(1), text)
                return text

            left = strip_tags(left)
            right = strip_tags(right)

            obj = {
                "messages": [
                    {"role": "user", "content": left},
                    {"role": "assistant", "content": right}
                ]
            }

            if not first:
                f_out.write(',\n')
            json.dump(obj, f_out, ensure_ascii=False)
            first = False
            written += 1
        f_out.write('\n]\n')

    print(f"Wrote {written} items to {OUTPUT}")

if __name__ == '__main__':
    main()
