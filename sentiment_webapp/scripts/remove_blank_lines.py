#!/usr/bin/env python3
"""Remove all blank/empty lines from text source files under a directory.
Creates a `.bak` backup for each modified file.
Usage:
  python remove_blank_lines.py -r /path/to/root --exts .py,.html,... --exclude models,*.keras
WARNING: This will remove ALL lines that are empty or contain only whitespace.
"""
import argparse
import os
import sys
from typing import Set
TEXT_EXTS = {'.py', '.html', '.css', '.js', '.md', '.txt', '.json', '.yml', '.yaml', '.ini'}
def should_exclude(dirpath, exclude_dirs):
    for ex in exclude_dirs:
        if not ex:
            continue
        if os.path.normpath(ex) in os.path.normpath(dirpath):
            return True
    return False
def process_file(path: str, dry_run: bool = False) -> bool:
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        new_lines = [ln for ln in lines if ln.strip() != '']
        if new_lines == lines:
            return True
        if dry_run:
            print(f"[DRY] Would update: {path} (removed {len(lines)-len(new_lines)} blank lines)")
            return True
        # backup
        with open(path + '.bak', 'w', encoding='utf-8') as f:
            f.writelines(lines)
        with open(path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print(f"[UPDATED] {path} (removed {len(lines)-len(new_lines)} blank lines)")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to process {path}: {e}")
        return False
def main():
    parser = argparse.ArgumentParser(description='Remove blank/empty lines from files')
    parser.add_argument('--root', '-r', default='.', help='Root directory to process')
    parser.add_argument('--exts', '-e', default=','.join(sorted(TEXT_EXTS)),
                        help='Comma-separated list of extensions to process')
    parser.add_argument('--exclude', '-x', default='models,*.keras,*.npz,static/uploads',
                        help='Comma-separated list of dirs/patterns to exclude')
    parser.add_argument('--dry-run', action='store_true', help='Do not write changes; just report')
    args = parser.parse_args()
    exts: Set[str] = { (e.strip().lower() if e.strip().startswith('.') else '.' + e.strip().lower()) for e in args.exts.split(',') }
    exclude_dirs = [p.strip() for p in args.exclude.split(',') if p.strip()]
    root = os.path.abspath(args.root)
    processed = 0
    failed = 0
    skipped = 0
    for dirpath, dirnames, filenames in os.walk(root):
        if any(part.startswith('.venv') or part == 'venv' for part in dirpath.split(os.sep)):
            continue
        if should_exclude(dirpath, exclude_dirs):
            skipped += len(filenames)
            dirnames[:] = []
            continue
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            _, ext = os.path.splitext(fname)
            ext = ext.lower()
            if ext not in exts:
                continue
            try:
                size = os.path.getsize(fpath)
                if size > 10 * 1024 * 1024:
                    print(f"[SKIP] {fpath} (large file)")
                    skipped += 1
                    continue
            except Exception:
                pass
            ok = process_file(fpath, dry_run=args.dry_run)
            if ok:
                processed += 1
            else:
                failed += 1
    print('\nSummary:')
    print(f'  Processed: {processed}')
    print(f'  Failed:    {failed}')
    print(f'  Skipped:   {skipped}')
if __name__ == '__main__':
    main()
