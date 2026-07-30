"""Fenster-Tagging: Dokument-Splitting und Facetten-Merge über Fenster.
Pure Funktionen, kein Netzwerk."""
from tagger_windowing import merge_window_facets, split_windows


def test_split_short_text_is_single_window():
    assert split_windows("kurz", 7000) == ["kurz"]


def test_split_covers_full_text_in_order():
    text = "a" * 15000
    windows = split_windows(text, 7000)
    assert len(windows) == 3
    assert "".join(windows) == text


def test_split_caps_window_count():
    text = "x" * 100000
    windows = split_windows(text, 1000, max_windows=10)
    assert len(windows) == 10


def test_merge_unions_and_orders_by_frequency():
    merged = merge_window_facets([
        {"schlagworte": ["duldung", "traumatisierung"], "herkunftsland": "syrien", "normen": ["§ 3 AsylG"]},
        {"schlagworte": ["traumatisierung", "attest"], "herkunftsland": "syrien", "normen": ["§ 3 AsylG", "Art. 3 EMRK"]},
        {"schlagworte": [], "herkunftsland": None, "normen": []},
    ])
    assert merged["schlagworte"][0] == "traumatisierung"  # 2x vor 1x
    assert set(merged["schlagworte"]) == {"traumatisierung", "duldung", "attest"}
    assert merged["herkunftsland"] == "syrien"
    assert merged["normen"][0] == "§ 3 AsylG"


def test_merge_caps_list_lengths():
    results = [{"schlagworte": [f"t{i}" for i in range(30)], "herkunftsland": None, "normen": [f"§ {i}" for i in range(30)]}]
    merged = merge_window_facets(results, max_themen=12, max_normen=12)
    assert len(merged["schlagworte"]) == 12
    assert len(merged["normen"]) == 12


def test_merge_empty_results_gives_empty_facets():
    assert merge_window_facets([]) == {"schlagworte": [], "herkunftsland": None, "normen": []}
