with open('serving/__main__.py', 'r') as f:
    content = f.read()

# Let's search for "laer_gamma" and print its surrounding context for all matches
import re

matches = [m.start() for m in re.finditer('laer_gamma', content)]
print(f"Found {len(matches)} occurrences of laer_gamma:")

for idx, pos in enumerate(matches):
    start = max(0, pos - 200)
    end = min(len(content), pos + 100)
    print(f"--- Occurrence {idx+1} ---")
    print(repr(content[start:end]))
