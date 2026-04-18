#!/usr/bin/env python3
"""
Annotate documents with a vocabulary warning at top and bottom.

Usage:
    python annotate_vocab_warning.py <file_or_glob> [<file_or_glob> ...]

The script:
  - Prepends a hyper-explicit H1 vocabulary warning to each file
  - Appends the same warning at the bottom (with an "END OF DOCUMENT" header)
  - Skips files that already contain the warning marker
  - Preserves the original file's content verbatim between the banners

Files affected:
  - Project memory at ~/.claude/projects/R--winrapids/memory/*.md
  - Skip-but-warn docs (session insights, prize problems, fintek-playground)
  - All campsites under R:\\winrapids\\campsites\\**\\*.md

For canonical vocabulary, see:
  - R:\\winrapids\\docs\\architecture\\vocabulary.md
  - R:\\winrapids\\docs\\architecture\\atoms-primitives-recipes.md
"""

import sys
import glob
import os

WARNING_MARKER = "VOCABULARY_WARNING_v1"

WARNING_TOP = """<!-- VOCABULARY_WARNING_v1 — do not remove this marker -->

# ⚠️ STOP — VOCABULARY WARNING — READ BEFORE PROCEEDING ⚠️

> **THIS DOCUMENT MAY CONTAIN OUTDATED VOCABULARY.**
>
> Tambear's vocabulary was LOCKED IN on 2026-04-17 with formal
> definitions. The terminology used in this document was current
> at the time of writing but may DIFFER from the locked vocabulary.
>
> **Do not assume any term in this document means what you think it
> means.** Words like *primitive*, *atom*, *recipe*, *method*,
> *specialist*, *operation*, *layer*, *kingdom*, *menu* may have
> meant something different at the time this document was written
> than they do in the current locked vocabulary.
>
> **Before relying on anything in this document:**
>
> 1. **Read the canonical vocabulary first** at:
>    `R:\\winrapids\\docs\\architecture\\vocabulary.md`
> 2. **Read the architecture decomposition** at:
>    `R:\\winrapids\\docs\\architecture\\atoms-primitives-recipes.md`
> 3. **Interpret this document's content through the locked lens.**
>    For every vocabulary term you encounter, ask: what does this
>    actually mean in current tambear? Use the "old term → locked
>    term" mapping table in `vocabulary.md`.
> 4. **QUESTION EVERYTHING.** Do not accept any vocabulary as
>    correct just because it sounds right or appears in this
>    document. The fact that a word is used here is NOT evidence
>    that the word's meaning here matches its current meaning.
>
> If you find inconsistencies between this document and the locked
> vocabulary, **the locked vocabulary in `vocabulary.md` is
> authoritative.** This document is a snapshot in time, not a
> current specification.
>
> Apparent agreement between this document and the locked vocabulary
> may be illusory — the same word may carry different meanings.
> CHECK THE MAPPING TABLE.

---

"""

WARNING_BOTTOM = """

---

<!-- VOCABULARY_WARNING_v1_END — do not remove this marker -->

# ⚠️ END OF DOCUMENT — VOCABULARY WARNING REPEATED ⚠️

> **REMINDER: Vocabulary in this document may be outdated.**
>
> Canonical vocabulary lives at:
> - `R:\\winrapids\\docs\\architecture\\vocabulary.md` (terminology)
> - `R:\\winrapids\\docs\\architecture\\atoms-primitives-recipes.md`
>   (architecture decomposition)
>
> **Do not trust vocabulary appearances. Question every term.**
> Map old language to the locked vocabulary BEFORE acting on the
> content of this document. The mapping table is in
> `vocabulary.md`.
>
> Words that may carry old meanings in this document:
> *primitive*, *atom*, *recipe*, *method*, *specialist*,
> *operation*, *layer*, *kingdom*, *menu*, *scatter*,
> *Layer 0/1/2/3/4*, *3-tier*, *9 truths*.
>
> If you arrived here from inside this document and skipped the
> top banner: GO BACK AND READ IT. The locked vocabulary is not
> a suggestion; it is the only correct interpretation of any
> tambear architecture document. Documents prior to 2026-04-17
> drift; trust the locked vocabulary, not the words in front of
> you.

"""


# Files that are the canonical sources of vocabulary truth — NEVER annotate these.
# The annotation would be self-referential (the file warns "see this file") and
# would erode the authority of the canonical doc. Paths are normalized to forward
# slashes for cross-platform comparison.
NEVER_ANNOTATE_BASENAMES = {
    "vocabulary.md",
    "atoms-primitives-recipes.md",
}


def is_canonical_source(path: str) -> bool:
    base = os.path.basename(path).lower()
    return base in NEVER_ANNOTATE_BASENAMES


def is_already_annotated(content: str) -> bool:
    return WARNING_MARKER in content


def annotate_file(path: str) -> str:
    """Return one of: 'annotated', 'skipped (already)', 'skipped (canonical)', 'skipped (not file)', 'error: <msg>'."""
    if not os.path.isfile(path):
        return "skipped (not a file)"
    if is_canonical_source(path):
        return "skipped (canonical source — never annotate)"
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception as e:
        return f"error reading: {e}"

    if is_already_annotated(content):
        return "skipped (already annotated)"

    new_content = WARNING_TOP + content + WARNING_BOTTOM

    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
    except Exception as e:
        return f"error writing: {e}"

    return "annotated"


def expand_args(args: list[str]) -> list[str]:
    expanded: list[str] = []
    for arg in args:
        if any(ch in arg for ch in "*?["):
            matches = glob.glob(arg, recursive=True)
            expanded.extend(matches)
        else:
            expanded.append(arg)
    return expanded


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 1

    files = expand_args(argv[1:])
    if not files:
        print("No files matched.", file=sys.stderr)
        return 1

    counts = {"annotated": 0, "skipped (already annotated)": 0,
              "skipped (not a file)": 0, "errors": 0}

    for path in files:
        result = annotate_file(path)
        if result == "annotated":
            counts["annotated"] += 1
            print(f"  annotated: {path}")
        elif result == "skipped (already annotated)":
            counts["skipped (already annotated)"] += 1
        elif result == "skipped (not a file)":
            counts["skipped (not a file)"] += 1
        elif result.startswith("skipped (canonical"):
            counts.setdefault("skipped (canonical source)", 0)
            counts["skipped (canonical source)"] += 1
            print(f"  skipped (canonical, will never annotate): {path}")
        else:
            counts["errors"] += 1
            print(f"  ERROR ({path}): {result}", file=sys.stderr)

    print("")
    print(f"Done. annotated={counts['annotated']} "
          f"already={counts['skipped (already annotated)']} "
          f"not-file={counts['skipped (not a file)']} "
          f"errors={counts['errors']}")
    return 0 if counts["errors"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
