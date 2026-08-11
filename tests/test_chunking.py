from ragcore.chunking import chunk_document, recursive_split, structure_split


def test_recursive_split_respects_size_and_overlap():
    text = "Sentence one. Sentence two. " * 40
    chunks = recursive_split(text, size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 260 for c in chunks)  # size + a small tail


def test_structure_split_attaches_section_headings():
    text = "# Alpha\n\nThe alpha section body.\n\n# Beta\n\nThe beta section body."
    pairs = structure_split(text, size=500, overlap=50)
    sections = {s for s, _ in pairs}
    assert "Alpha" in sections and "Beta" in sections


def test_chunk_document_metadata():
    text = "# Warranty\n\nThe widget has a two-year warranty. Returns within 30 days."
    chunks = chunk_document("doc1", text, strategy="structure")
    assert chunks
    c = chunks[0]
    assert c.doc_id == "doc1"
    assert c.content_hash  # auto-computed
    assert c.ordinal == 0
    assert c.section == "Warranty"


def test_empty_document_yields_nothing():
    assert chunk_document("d", "   ") == []
