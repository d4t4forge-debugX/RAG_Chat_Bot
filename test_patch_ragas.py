from patch_ragas import NEW, OLD, apply_patch


# verifies the original broken import lines are rewritten to the try/except version
def test_unpatched_source_gets_patched():
    new_source, status = apply_patch("import x\n" + OLD + "import y\n")
    assert status == "patched"
    assert NEW in new_source
    assert "import x" in new_source and "import y" in new_source


# verifies running the patch twice changes nothing the second time
def test_patching_is_idempotent():
    once, _ = apply_patch("import x\n" + OLD)
    twice, status = apply_patch(once)
    assert status == "already_patched"
    assert twice == once


# verifies a file without the expected lines is refused instead of being modified
def test_unrecognised_source_is_left_alone():
    source = "something else entirely\n"
    new_source, status = apply_patch(source)
    assert status == "unexpected"
    assert new_source == source