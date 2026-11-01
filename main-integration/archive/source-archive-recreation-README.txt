Recreating the four frozen source archives

The verified exact recipe is recreate-exact-source-archives.py. Keep it beside
recreate-source-archives.py (its identity/checking helper) and
source-archive-recreation-inputs.json. Neither script edits a source checkout,
candidate tree, original archive, or prior build/run evidence. Output must be
a new or empty directory. The repository must contain all four frozen objects.

From the original host workspace root:

  work/.venv/bin/python work/main-integration/recreate-exact-source-archives.py \
    --repo work/vllm \
    --output work/main-integration/my-fresh-source-archives

Use an existing Python 3.12+ virtual environment on another host; replace the
paths as appropriate. A Git executable supporting archive --mtime and a gzip
executable must be available. --reference-dir is optional and was additionally
used in the recorded validation to compare against the original local archives
byte-for-byte. It is not needed to check the published expected SHA256 values.

The upstream commit alone does not contain the candidate trees. If starting
from a fresh repository, reconstruct each candidate with its published
{erf,bf16,openmp}-frozen.patch on e0d00e4b8eb0c89ec30921d0695cfdd409e9c211,
using a disposable checkout/index, then verify git write-tree matches the
corresponding frozen tree in source-archive-recreation-inputs.json. The script
refuses to proceed when a required commit/tree object is unavailable or wrong.
Do not substitute a later PR head or current main.

Tar identities
variant  tree-ish type  entry mtime   global PAX comment
base     commit         1793502281    e0d00e4b8eb0c89ec30921d0695cfdd409e9c211
erf      tree           1793507340    absent
bf16     tree           1793507352    absent
openmp   tree           1793507394    absent
Each prefix is main-source-{variant}/. Exact refs, archive sizes and SHA256
values are in the inputs manifest. Every tar entry's timestamp is fixed using
git archive --mtime=@{timestamp}. Gzip header timestamps are separately zero.

Compression difference discovered and verified
The original baseline uses Apple gzip -n: header FLG=0, XFL=0, OS=3.
The original three candidates have FLG=8, XFL=2, OS=255 and embed the filename
main-source-{variant}.tar. Their exact reproduction is Python gzip.GzipFile
with that filename, compresslevel=9 and mtime=0. Thus a uniform gzip -n command
cannot reproduce all four original compressed byte streams.

The initial recreate-source-archives.py attempt intentionally remains unchanged.
Its driver exited 1: baseline matched exactly; all three candidate compressed
hashes differed, although all four uncompressed tar hashes and ordered member
content/mode hashes matched. These results are retained in
source-archive-recreation-{driver,results}.json and source-archive-recreation-run.txt.

The separate exact-recipe attempt then exited 0. All four expected compressed
hashes, sizes, uncompressed tar hashes, member-content hashes, mtimes/PAX and
actual byte comparisons to the originals passed. See source-archive-exact-
{driver,results}.json and source-archive-exact-run.txt. Tested versions:
git version 2.50.1 (Apple Git-155); Apple gzip 475; Python 3.12.14;
compile-time and runtime zlib 1.2.12. Both scripts passed Ruff checks/format.

Portability and independent content fallback
Other gzip/zlib/Git versions can produce different compressed bytes. Do not
claim exact archive reproduction unless exact_archive_match is true. The
script retains the exact generated command, tool versions and observed hashes.
uncompressed_tar_match separately proves equality of the decoded tar byte
stream. unpacked_content_and_mode_match checks the inputs manifest's ordered
SHA256 summary of each member's path, type, permissions, symlink target, size
and regular-file content SHA256. This deliberately excludes timestamps,
uid/gid, owner/group names and PAX layout, and is not an archive-byte claim.
No archive is extracted to perform these comparisons. A mismatch still exits
nonzero; inspect which identity layer differs rather than silently accepting it.

Only the small scripts, inputs, README and raw text/JSON validation results are
proposed for publication by source-archive-recreation-publish.json. The two
fresh output directories, all new/original archives and raw temporary tar data
are excluded. The original archives were never overwritten.
