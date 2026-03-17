import os
import csv
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

args = sys.argv[1:]

if any(a in ('-h', '--help', 'help') for a in args):
    script = os.path.basename(sys.argv[0])
    print(f"""Usage: {script} [ROOT ...] [OPTIONS]

Recursively scan one or more directories for CSV files and count occurrences
of each value in the 'label' column. Output can be grouped by label, file
path, or ID prefix.

Arguments:
  ROOT                  One or more directories to scan recursively.
                        Defaults to the directory containing this script.

Options:
  pdirs                 Under each label, list the files it appears in
                        along with per-file counts.
  byid                  Group results by patient/subject ID, derived from
                        the filename prefix before the first underscore
                        (e.g. 'chb01' from 'chb01_03.csv').
  exclude=TERM[,...]    Skip CSV files whose names contain any of the given
                        terms (case-insensitive, comma-separated).

Examples:
  {script}
  {script} /data/chbmit
  {script} /data/chbmit pdirs
  {script} /data/chbmit byid
  {script} /data/chbmit byid exclude=seizure
  {script} /data/chbmit pdirs exclude=seizure,debug,tmp
  {script} /data/a /data/b byid exclude=seizure

Notes:
  - Exclusions match against filenames only, not directory names.
  - Lines beginning with '#' are treated as comments and skipped.
  - Only .csv files are scanned.
""")
    sys.exit(0)



show_dirs = 'pdirs' in args
by_id = 'byid' in args

exclude_terms = []
for a in args:
    if a.startswith('exclude='):
        value = a.split('=', 1)[1].strip()
        if value:
            exclude_terms.extend(
                term.strip().lower()
                for term in value.split(',')
                if term.strip()
            )

roots = [
    a for a in args
    if a not in ('pdirs', 'byid') and not a.startswith('exclude=')
] or [os.path.dirname(os.path.abspath(__file__))]

roots = [os.path.realpath(r) for r in roots]

for root in roots:
    if not os.path.isdir(root):
        sys.exit(f"Error: '{root}' is not a valid directory.")

def should_exclude(filename):
    name = os.path.basename(filename).lower()
    return any(term in name for term in exclude_terms)

def extract_labels(filepath):
    with open(filepath, encoding='utf-8') as f:
        rows = csv.DictReader(line for line in f if not line.startswith('#'))
        return filepath, Counter(row['label'].strip() for row in rows if row.get('label'))

def strip_root(filepath):
    for root in roots:
        if filepath.startswith(root):
            return filepath[len(root):].lstrip(os.sep)
    return filepath

def get_id(filepath):
    return os.path.basename(filepath).split('_')[0]

csv_files = list({
    os.path.realpath(os.path.join(dirpath, f))
    for root in roots
    for dirpath, _, filenames in os.walk(root)
    for f in filenames
    if f.endswith('.csv') and not should_exclude(f)
})

label_paths = {}  # { label: { filepath: count } }
id_data = {}      # { id: { labels: Counter, files: set } }

with ThreadPoolExecutor() as executor:
    futures = [executor.submit(extract_labels, f) for f in csv_files]
    for filepath, counts in (f.result() for f in as_completed(futures)):
        for label, count in counts.items():
            label_paths.setdefault(label, {})[filepath] = count
        pid = get_id(filepath)
        entry = id_data.setdefault(pid, {'labels': Counter(), 'files': set()})
        entry['labels'].update(counts)
        entry['files'].add(filepath)

total_files = len(csv_files)
print(f"\n{len(label_paths)} unique label{'s' if len(label_paths) != 1 else ''} across {total_files} csv file{'s' if total_files != 1 else ''}\n")

if by_id:
    unique_ids = len(id_data)
    print(f"{unique_ids} unique id{'s' if unique_ids != 1 else ''} across {total_files} csv file{'s' if total_files != 1 else ''}\n")

    for pid in sorted(id_data):
        labels = id_data[pid]['labels']
        files = id_data[pid]['files']
        total = sum(labels.values())
        print(f"{pid} ({len(files)} file{'s' if len(files) != 1 else ''}, {total} total rows):")
        for label in sorted(labels):
            print(f"  {label}: {labels[label]}")
elif show_dirs:
    for label in sorted(label_paths):
        files = label_paths[label]
        print(f"{label} ({sum(files.values())} total across {len(files)} file{'s' if len(files) != 1 else ''}):")
        for path in sorted(files):
            print(f"  {strip_root(path)} ({files[path]})")
else:
    for label in sorted(label_paths):
        files = label_paths[label]
        print(f"  {label} ({sum(files.values())} total across {len(files)} file{'s' if len(files) != 1 else ''})")
