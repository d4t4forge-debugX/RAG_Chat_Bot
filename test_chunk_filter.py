from chunk_filter import is_noise_chunk

PROSE = (
    "Supervised learning is a type of machine learning where the training set you feed "
    "to the algorithm includes the desired solutions, called labels. A typical task is "
    "classification, such as a spam filter trained on many example emails."
)

TOC = (
    "Table of Contents\n"
    "Preface. . . . . . . . . . . . . . . . . . . . . . . . xv\n"
    "Part I. The Fundamentals of Machine Learning\n"
    "1. The Machine Learning Landscape. . . . . . . . . . . . . . . . . 3\n"
    "What Is Machine Learning?. . . . . . . . . . . . . . . . . . . . . 4\n"
    "Why Use Machine Learning?. . . . . . . . . . . . . . . . . . . . . 5\n"
)

INDEX = (
    "reconstruction loss, 397, 570\n"
    "reconstruction pre-images, 228\n"
    "reconstructions, 570\n"
    "Rectified linear unit (ReLU), 281\n"
    "recurrent neural networks, 497\n"
    "regression, 41, 105, 135, 142, 158\n"
    "regularization, 26, 134\n"
)

TABLE = (
    "Table 3: Variations on the Transformer architecture. Unlisted values are identical to those of the base model.\n"
    "N dmodel dff h dk dv Pdrop PPL BLEU params\n"
    "base 6 512 2048 8 64 64 0.1 0.1 100K 4.92 25.8 65\n"
    "1 512 512 5.29 24.9\n"
    "4 128 128 5.00 25.5\n"
    "16 32 32 4.91 25.8\n"
    "32 16 16 5.01 25.4\n"
    "2 5.75 24.5 36\n"
    "4 4.66 26.0 50\n"
    "8 4.88 25.5 80\n"
    "1024 4.66 26.0 168\n"
    "4096 4.75 26.2 90\n"
    "0.0 5.77 24.6\n"
    "0.2 4.95 25.5\n"
    "0.1 4.67 25.3\n"
)

CODE = (
    ">>> X_new_b = np.c_[np.ones((2, 1)), X_new]  # add x0 = 1 to each instance\n"
    ">>> y_predict = theta_best.dot(X_new_b)\n"
    ">>> y_predict\n"
    "array([[ 4.21509616],\n"
    "       [ 9.75532293]])"
)

# the first lines of the real Table 4 chunk from the Transformer paper, which earlier versions of the filter wrongly flagged
REAL_PAPER_TABLE = (
    "Table 4: The Transformer generalizes well to English constituency parsing (Results are on Section 23\n"
    "of WSJ)\n"
    "Parser Training WSJ 23 F1\n"
    "Vinyals & Kaiser el al. (2014) [37] WSJ only, discriminative 88.3\n"
    "Petrov et al. (2006) [29] WSJ only, discriminative 90.4\n"
    "Zhu et al. (2013) [40] WSJ only, discriminative 90.4\n"
    "Dyer et al. (2016) [8] WSJ only, discriminative 91.7\n"
    "Transformer (4 layers) WSJ only, discriminative 91.3\n"
    "Zhu et al. (2013) [40] semi-supervised 91.3\n"
    "Huang & Harper (2009) [14] semi-supervised 91.3\n"
    "McClosky et al. (2006) [26] semi-supervised 92.1\n"
)

# a few real-style index lines (ranges and comma-separated references)
REAL_STYLE_INDEX = (
    "bagging and pasting, 192-196\n"
    "benefits of, 74\n"
    "best uses of, 191\n"
    "boosting, 199-208\n"
    "hard clustering, 240\n"
    "hard margin classification, 154\n"
    "hard voting classifiers, 190\n"
    "harmonic mean, 90\n"
)


# verifies a part-divider fragment is treated as noise
def test_short_divider_is_noise():
    assert is_noise_chunk("PART I\nThe Fundamentals of\nMachine Learning") is True


# verifies ordinary explanatory prose is kept
def test_normal_prose_is_kept():
    assert is_noise_chunk(PROSE) is False


# verifies a table-of-contents page with dot leaders is noise
def test_table_of_contents_is_noise():
    assert is_noise_chunk(TOC) is True


# verifies an index page of short text lines ending in page numbers is noise
def test_index_page_is_noise():
    assert is_noise_chunk(INDEX) is True


# verifies prose that merely contains a few numbers is not mistaken for an index
def test_prose_with_some_numbers_is_kept():
    text = PROSE + " In the book's example the model reached 98% accuracy after 10 epochs on 60,000 images."
    assert is_noise_chunk(text) is False


# verifies a numeric results table is kept (this was a false positive in the first dry run)
def test_numeric_table_is_kept():
    assert is_noise_chunk(TABLE) is False


# verifies a code listing with printed output is kept (also a false positive in the first dry run)
def test_code_listing_is_kept():
    assert is_noise_chunk(CODE) is False


# verifies the real paper table (decimal scores) is kept
def test_real_paper_table_is_kept():
    assert is_noise_chunk(REAL_PAPER_TABLE) is False


# verifies real-style index entries with page ranges are still caught
def test_real_style_index_is_noise():
    assert is_noise_chunk(REAL_STYLE_INDEX) is True