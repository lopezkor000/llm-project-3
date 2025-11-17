#!/usr/bin/env python3
import csv
import json
import sys
import html
from pathlib import Path

INPUT = Path('michael_dataset.csv')
OUTPUT = Path('michael_dataset.json')

USER_FIELD = 'input'     # use ONLY this column for user prompt
ASSIST_FIELD = 'output'  # assistant reply

def main():
    if not INPUT.exists():
        print(f"Input CSV not found: {INPUT}", file=sys.stderr)
        sys.exit(1)

    try:
        csv.field_size_limit(sys.maxsize)
    except Exception:
        pass

    written = 0
    with INPUT.open('r', encoding='utf-8', newline='') as f_in, OUTPUT.open('w', encoding='utf-8') as f_out:
        reader = csv.DictReader(f_in)
        if USER_FIELD not in reader.fieldnames or ASSIST_FIELD not in reader.fieldnames:
            print(f"Unexpected headers. Found: {reader.fieldnames}", file=sys.stderr)
            sys.exit(2)

        f_out.write('[\n')
        first = True
        for row in reader:
            user_content = html.unescape((row.get(USER_FIELD) or '').strip())
            assistant_content = html.unescape((row.get(ASSIST_FIELD) or '').strip())
            if not user_content or not assistant_content:
                continue
            obj = {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content}
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
