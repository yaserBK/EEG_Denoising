import os
import csv
import sys
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

args = sys.argv[1:]

if any(a in ('-h', '--help', 'help') for a in args):
    script = os.path.basename(sys.argv[0])
    print(f"""Usage: {script} [ROOT ...] [OPTIONS]

Recursively scan one or more directories for CSV files, count occurrences
of each value in the 'label' column, and write one JSON file per ID plus a
combined JSON file for the full scan. EDF files and _seiz CSV files in the
same directory are matched to their CSV by base name.

Arguments:
  ROOT                  One or more directories to scan recursively.
                        Defaults to the directory containing this script.

Options:
  pdirs                 Under each label, list the files it appears in
                        along with per-file counts.
  byid                  Group results by patient/subject ID, derived from
                        the filename prefix before the first underscore
                        (e.g. 'chb01' from 'chb01_03.csv').
  listids               Print only the unique IDs found, with no other data.
  exclude=TERM[,...]    Skip CSV files whose names contain any of the given
                        terms (case-insensitive, comma-separated).
  out=PATH              Directory to write JSON files to.
                        Defaults to the current working directory.
  quiet                 Suppress terminal output; only write JSON files.

JSON output:
  One file is written per ID (e.g. 'chb01.json') plus a combined file
  (e.g. 'chbmit_all.json') containing all IDs keyed by ID.

  Per-ID files have a summary scoped to that ID only:
    - total_edf_files   Number of EDF files for this ID
    - total_csv_files   Number of CSV files for this ID
    - total_duration    Sum of durations for this ID
    - total_labels      Number of unique labels for this ID
    - label_totals      Label counts for this ID only

  has_seiz is set at both the ID and recording level:
    - ID level          True if any recording for this ID has a _seiz CSV
    - Recording level   True if this specific recording has a _seiz CSV

  The combined file has a global summary across all IDs:
    - total_edf_files   Total EDF files across all IDs
    - total_csv_files   Total CSV files across all IDs
    - total_duration    Total duration across all IDs
    - total_labels      Number of unique labels across all IDs
    - label_totals      Label counts across all IDs

Examples:
  {script}
  {script} /data/chbmit
  {script} /data/chbmit quiet
  {script} /data/chbmit out=/results
  {script} /data/chbmit byid exclude=seizure
  {script} /data/a /data/b byid exclude=seizure,debug

Notes:
  - _seiz CSV files are matched to their main CSV by base name and listed
    as seiz_csv in the recording block. They are not counted in total_csv_files.
  - Exclusions match against filenames only, not directory names.
  - Lines beginning with '#' are treated as comments and skipped.
  - Only .csv files are scanned.
""")
    sys.exit(0)


show_dirs = 'pdirs' in args
by_id     = 'byid' in args
list_ids  = 'listids' in args
quiet     = 'quiet' in args

out_dir = '.'
for a in args:
    if a.startswith('out='):
        out_dir = a.split('=', 1)[1].strip()

if not os.path.isdir(out_dir):
    sys.exit(f"Error: output directory '{out_dir}' does not exist.")

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
    if a not in ('pdirs', 'byid', 'listids', 'quiet')
    and not a.startswith('exclude=')
    and not a.startswith('out=')
] or [os.path.dirname(os.path.abspath(__file__))]

roots = [os.path.realpath(r) for r in roots]
roots = sorted(set(roots))
roots = [
    r for r in roots
    if not any(r.startswith(other + os.sep) for other in roots if other != r)
]

for root in roots:
    if not os.path.isdir(root):
        sys.exit(f"Error: '{root}' is not a valid directory.")

def should_exclude(filename):
    name = os.path.basename(filename).lower()
    return any(term in name for term in exclude_terms)

def parse_header(filepath):
    meta = {}
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            if not line.startswith('#'):
                break
            line = line.lstrip('#').strip()
            if '=' in line:
                key, _, val = line.partition('=')
                meta[key.strip()] = val.strip()
    return meta

def parse_csv(filepath):
    meta = parse_header(filepath)
    duration = None
    if 'duration' in meta:
        try:
            duration = float(meta['duration'].split()[0])
        except ValueError:
            pass

    label_data = defaultdict(lambda: {'count': 0, 'segments': []})

    with open(filepath, encoding='utf-8') as f:
        rows = csv.DictReader(line for line in f if not line.startswith('#'))
        for row in rows:
            label = row.get('label', '').strip()
            if not label:
                continue

            start      = None
            stop       = None
            channel    = None
            confidence = None

            try:
                start = float(row['start_time'].strip())
            except (KeyError, ValueError):
                pass
            try:
                stop = float(row['stop_time'].strip())
            except (KeyError, ValueError):
                pass
            if 'channel' in row:
                channel = row['channel'].strip() or None
            if 'confidence' in row:
                try:
                    confidence = float(row['confidence'].strip())
                except (KeyError, ValueError):
                    pass

            label_data[label]['count'] += 1
            label_data[label]['segments'].append({
                'start':      start,
                'stop':       stop,
                'channel':    channel,
                'confidence': confidence
            })

    return duration, {k: dict(v) for k, v in sorted(label_data.items())}

def extract_main(filepath):
    duration, label_data = parse_csv(filepath)
    return filepath, label_data, duration

def extract_seiz(filepath):
    _, label_data = parse_csv(filepath)
    return filepath, label_data

def strip_root(filepath, root):
    if filepath.startswith(root):
        return filepath[len(root):].lstrip(os.sep)
    return filepath

def get_id(filepath):
    return os.path.basename(filepath).split('_')[0]

def is_seiz_csv(filename):
    return os.path.splitext(filename)[0].endswith('_seiz')

def seiz_base(filename):
    return os.path.splitext(filename)[0][:-len('_seiz')]

def build_id_summary(entry, edf_index):
    id_edf_count  = sum(1 for base in entry['recordings'] if edf_index.get(base) is not None)
    unique_labels = set(entry['label_counts'].keys()) | set(entry['seiz_label_counts'].keys())
    label_totals  = dict(sorted({
        **{l: c for l, c in entry['label_counts'].items()},
        **{l: entry['label_counts'].get(l, 0) + entry['seiz_label_counts'].get(l, 0)
           for l in entry['seiz_label_counts']
           if l not in entry['label_counts']}
    }.items()))
    return {
        "total_edf_files": id_edf_count,
        "total_csv_files": len(entry['files']),
        "total_duration":  entry['total_duration'],
        "total_labels":    len(unique_labels),
        "label_totals":    label_totals
    }

def build_id_payload(pid, entry, edf_index):
    has_seiz = any(rec['seiz_csv'] is not None for rec in entry['recordings'].values())
    return {
        "id":          pid,
        "has_seiz":    has_seiz,
        "summary":     build_id_summary(entry, edf_index),
        "total_rows":  entry['total_rows'],
        "labels":      dict(sorted(entry['label_counts'].items())),
        "seiz_labels": dict(sorted(entry['seiz_label_counts'].items())),
        "recordings":  {
            base: rec
            for base, rec in sorted(entry['recordings'].items())
        }
    }

def build_global_summary(id_data, label_paths, edf_index):
    return {
        "total_edf_files": len(edf_index),
        "total_csv_files": sum(len(e['files']) for e in id_data.values()),
        "total_duration":  sum(e['total_duration'] for e in id_data.values()),
        "total_labels":    len(label_paths),
        "label_totals":    {
            label: sum(files.values())
            for label, files in sorted(label_paths.items())
        }
    }


for root in roots:
    csv_files = list({
        os.path.realpath(os.path.join(dirpath, f))
        for dirpath, _, filenames in os.walk(root)
        for f in filenames
        if f.endswith('.csv') and not is_seiz_csv(f) and not should_exclude(f)
    })

    seiz_csv_files = {
        seiz_base(f): os.path.realpath(os.path.join(dirpath, f))
        for dirpath, _, filenames in os.walk(root)
        for f in filenames
        if f.endswith('.csv') and is_seiz_csv(f)
    }

    edf_index = {}
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.endswith('.edf'):
                base = os.path.splitext(f)[0]
                edf_index[base] = strip_root(
                    os.path.realpath(os.path.join(dirpath, f)), root
                )

    seiz_data = {}
    with ThreadPoolExecutor() as executor:
        seiz_futures = {
            executor.submit(extract_seiz, fp): base
            for base, fp in seiz_csv_files.items()
        }
        for future in as_completed(seiz_futures):
            base = seiz_futures[future]
            _, label_data = future.result()
            seiz_data[base] = label_data

    label_paths = {}
    id_data     = {}

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(extract_main, f) for f in csv_files]
        for filepath, label_data, duration in (f.result() for f in as_completed(futures)):
            counts = {label: v['count'] for label, v in label_data.items()}
            for label, count in counts.items():
                label_paths.setdefault(label, {})[filepath] = count

            pid  = get_id(filepath)
            base = os.path.splitext(os.path.basename(filepath))[0]
            rel  = strip_root(filepath, root)

            entry = id_data.setdefault(pid, {
                'label_counts':      Counter(),
                'seiz_label_counts': Counter(),
                'total_rows':        0,
                'total_duration':    0.0,
                'files':             set(),
                'recordings':        {}
            })

            rec_seiz_data = seiz_data.get(base, {})
            seiz_counts   = {label: v['count'] for label, v in rec_seiz_data.items()}

            entry['label_counts'].update(counts)
            entry['seiz_label_counts'].update(seiz_counts)
            entry['total_rows']     += sum(counts.values())
            entry['total_duration'] += duration if duration is not None else 0.0
            entry['files'].add(filepath)
            entry['recordings'][base] = {
                'csv':         rel,
                'edf':         edf_index.get(base),
                'seiz_csv':    strip_root(seiz_csv_files[base], root) if base in seiz_csv_files else None,
                'has_seiz':    base in seiz_csv_files,
                'duration':    duration,
                'labels':      label_data,
                'seiz_labels': rec_seiz_data
            }

    # --- terminal output ---
    if not quiet:
        total_csv_files = len(csv_files)
        print(f"\n{len(label_paths)} unique label{'s' if len(label_paths) != 1 else ''} across {total_csv_files} csv file{'s' if total_csv_files != 1 else ''}\n")

        if list_ids:
            for pid in sorted(id_data):
                print(pid)
        elif by_id:
            unique_ids = len(id_data)
            print(f"{unique_ids} unique id{'s' if unique_ids != 1 else ''} across {total_csv_files} csv file{'s' if total_csv_files != 1 else ''}\n")
            for pid in sorted(id_data):
                label_counts = id_data[pid]['label_counts']
                files        = id_data[pid]['files']
                total        = id_data[pid]['total_rows']
                dur          = id_data[pid]['total_duration']
                has_seiz     = any(r['has_seiz'] for r in id_data[pid]['recordings'].values())
                print(f"{pid} ({len(files)} file{'s' if len(files) != 1 else ''}, {total} total rows, {dur:.1f}s, has_seiz={has_seiz}):")
                for label in sorted(label_counts):
                    print(f"  {label}: {label_counts[label]}")
        elif show_dirs:
            for label in sorted(label_paths):
                files = label_paths[label]
                print(f"{label} ({sum(files.values())} total across {len(files)} file{'s' if len(files) != 1 else ''}):")
                for path in sorted(files):
                    print(f"  {strip_root(path, root)} ({files[path]})")
        else:
            for label in sorted(label_paths):
                files = label_paths[label]
                print(f"  {label} ({sum(files.values())} total across {len(files)} file{'s' if len(files) != 1 else ''})")

    # --- JSON output: one file per ID ---
    combined = {}
    for pid, entry in sorted(id_data.items()):
        payload = build_id_payload(pid, entry, edf_index)
        combined[pid] = payload

        out_path = os.path.join(out_dir, f"{pid}.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)

        if not quiet:
            print(f"  JSON written to {out_path}")

    # --- combined JSON output ---
    root_name     = os.path.basename(root.rstrip(os.sep)) or 'root'
    combined_path = os.path.join(out_dir, f"{root_name}_all.json")
    combined_out  = {
        "summary": build_global_summary(id_data, label_paths, edf_index),
        "subjects":     combined
    }
    with open(combined_path, 'w', encoding='utf-8') as f:
        json.dump(combined_out, f, indent=2)

    if not quiet:
        print(f"\n  Combined JSON written to {combined_path}\n")
