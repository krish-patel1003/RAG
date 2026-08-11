"""Registry + indexer semantics: the operations tutorials skip."""

from ragcore.indexer import Indexer


def test_content_hash_skips_unchanged(rag_env):
    ix = Indexer()
    ix.index_document("d1", "The widget has a two-year warranty.")
    rep = ix.index_document("d1", "The widget has a two-year warranty.")
    assert rep["skipped"] is True
    assert rep["reason"] == "unchanged_content"


def test_reindex_deletes_old_chunks(rag_env):
    ix = Indexer()
    ix.index_document("d1", "Original body about warranties and returns.")
    ids_before = ix.store.active_chunk_ids("d1", ix.store.get_current_index_version())
    ix.index_document("d1", "Completely different body about shipping and delivery times.")
    ids_after = ix.store.active_chunk_ids("d1", ix.store.get_current_index_version())
    # Old chunk ids must be gone (delete-then-insert, not in-place update).
    assert set(ids_before).isdisjoint(set(ids_after))
    assert ix.store.stats()["active_docs"] == 1


def test_delete_document_hides_it(rag_env):
    ix = Indexer()
    ix.index_document("d1", "Some content that will be deleted.")
    ix.delete_document("d1")
    assert ix.store.stats()["active_docs"] == 0
    assert ix.store.active_chunk_ids("d1", ix.store.get_current_index_version()) == []


def test_alias_swap_promotes_shadow_index(rag_env):
    ix = Indexer()
    ix.index_document("d1", "Live content in version one.")
    assert ix.store.get_current_index_version() == "v1"
    # Build a shadow index under v2 without touching the live alias.
    ix.rebuild_shadow([("d1", "Rebuilt content in version two.")], new_version="v2")
    assert ix.store.get_current_index_version() == "v1"  # not yet promoted
    ix.promote("v2", gc_old=True)
    assert ix.store.get_current_index_version() == "v2"
