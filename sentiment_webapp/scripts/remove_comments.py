"""Remove comments from source files under a directory.
- Python files: use the tokenize module to safely remove `COMMENT` tokens.
- Other text files (.html, .css, .js, .md, .txt, .json, .yml, .ini): apply heuristic regex removals.
Backups are created with a `.bak` suffix before overwriting.
"""
import argparse
import io
import os
import re
import sys
import tokenize
TEXT_EXTS = {'.html', '.css', '.js', '.md', '.txt', '.json', '.yml', '.yaml', '.ini'}
def remove_comments_python(path: str) -> bool:
    try:
        with open(path, 'rb') as f:
            src = f.read()
        tokens = list(tokenize.tokenize(io.BytesIO(src).readline))
        new_tokens = []
        for tok in tokens:
            if tok.type == tokenize.COMMENT:
                continue
            new_tokens.append(tok)
        new_src = tokenize.untokenize(new_tokens)
        if isinstance(new_src, bytes):
            new_src = new_src.decode('utf-8')
        with open(path + '.bak', 'wb') as f:
            f.write(src)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_src)
        return True
    except Exception as e:
        print(f"[ERROR] Python comment removal failed for {path}: {e}")
        return False
def remove_comments_text(path: str) -> bool:
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        original = content
        content = re.sub(r'<!--.*?-->', '', content, flags=re.S)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.S)
        content = re.sub(r'(^|\s)//[^\n]*', r'\1', content)
        content = re.sub(r'^[ \t]*#.*\n?', '', content, flags=re.M)
        if content == original:
            return True
        with open(path + '.bak', 'w', encoding='utf-8') as f:
            f.write(original)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"[ERROR] Text comment removal failed for {path}: {e}")
        return False
def should_exclude(dirpath, exclude_dirs):
    for ex in exclude_dirs:
        if not ex:
            continue
        if os.path.normpath(ex) in os.path.normpath(dirpath):
            return True
    return False
def main():
    parser = argparse.ArgumentParser(description='Remove comments from source files')
    parser.add_argument('--root', '-r', default='.', help='Root directory to process')
    parser.add_argument('--exts', '-e', default='.py,.html,.css,.js,.md,.txt,.json,.yml,.yaml,.ini',
                        help='Comma-separated list of extensions to process')
    parser.add_argument('--exclude', '-x', default='models,*.keras,*.npz,static/uploads',
                        help='Comma-separated list of directories or patterns to exclude')
    parser.add_argument('--dry-run', action='store_true', help='Do not write changes; just report')
    args = parser.parse_args()
    exts = {e.strip().lower() if e.strip().startswith('.') else ('.' + e.strip().lower()) for e in args.exts.split(',')}
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
                if os.path.getsize(fpath) > 5 * 1024 * 1024:             
                    print(f"[SKIP] {fpath} (large file)")
                    skipped += 1
                    continue
            except Exception:
                pass
            print(f"[PROCESS] {fpath}")
            if args.dry_run:
                processed += 1
                continue
            ok = False
            if ext == '.py':
                ok = remove_comments_python(fpath)
            else:
                ok = remove_comments_text(fpath)
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
