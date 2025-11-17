#!/usr/bin/env python3
import csv
import json
import sys
import html
from pathlib import Path

INPUT = Path('dwight_dataset.csv')
OUTPUT = Path('dwight_dataset.json')

LIKELY_USER_FIELDS = ['instruction', 'prompt']
LIKELY_INPUT_FIELDS = ['input', 'context']
LIKELY_ASSIST_FIELDS = ['output', 'completion', 'response']

def pick_field(row, candidates):
    for c in candidates:
        for k in row.keys():
            if k.lower() == c:
                return k
    return None

def build_user_content_from_input(row, input_key):
    # Per requirement: ONLY the `input` column should be used for the user prompt
    if not input_key:
        return ''
    return html.unescape((row.get(input_key) or '').strip())

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
        # Flexible header detection
        if not reader.fieldnames:
            print("CSV has no headers.", file=sys.stderr)
            sys.exit(2)

        # Try to find standard fields; if not found, fallback to first two columns
        sample_row = next(reader, None)
        if sample_row is None:
            f_out.write('[]\n')
            print(f"Wrote 0 items to {OUTPUT}")
            return
        # Rewind by reopening to iterate from start
        f_in.seek(0)
        reader = csv.DictReader(f_in)

        # We intentionally ignore instruction/prompt for user content; only `input` is used
        user_key = None
        input_key = pick_field(sample_row, LIKELY_INPUT_FIELDS)
        assist_key = pick_field(sample_row, LIKELY_ASSIST_FIELDS)

        if not assist_key:
            # Fallback mapping: last column as assistant
            assist_key = reader.fieldnames[-1]
        # Do NOT fallback to instruction/other columns for user; if input missing we skip row

        f_out.write('[\n')
        first = True
        for row in reader:
            user_content = build_user_content_from_input(row, input_key)
            assistant_content = html.unescape((row.get(assist_key) or '').strip())
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
