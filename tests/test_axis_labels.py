"""Tests for reading tick labels and fitting the axis scale (trace_digitiser.axis_labels)."""

import pytest

from trace_digitiser.axis_labels import (
    TickLabel,
    _combine_power,
    _interpret,
    _readings_that_fit,
    _rungs,
    calibrate_axis,
    find_tick_labels,
    fit_axis,
    parse_number,
)
from trace_digitiser.frames import darkness_map
from trace_digitiser.io import load_image
from trace_digitiser.panel_detection import find_plot_panels

from .conftest import needs_published_figure, needs_tesseract


def values(text):
    return [v for v, _ in parse_number(text)]


class TestParseNumber:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("300", [300.0]),
            ("0", [0.0]),
            ("-25", [-25.0]),
            ("−25", [-25.0]),  # typographic minus
            ("0.5", [0.5]),
            ("1.0", [1.0]),
            (".5", [0.5]),
            ("300.", [300.0]),
            ("1,000", [1000.0]),
            ("12,500", [12500.0]),
            ("0,5", [0.5]),  # decimal comma
            ("1,25", [1.25]),
            ("+10", [10.0]),
            ("", []),
            ("-", []),
            (".", []),
            ("abc", []),
        ],
    )
    def test_plain(self, text, expected):
        assert values(text) == pytest.approx(expected)

    def test_lost_decimal_point(self):
        # "05" cannot be an integer tick label: the point was lost.
        assert parse_number("05") == [(0.5, False), (5.0, True)]
        assert parse_number("025") == [(0.25, False), (25.0, True)]
        assert values("00") == [0.0]

    def test_power_of_ten(self):
        assert _combine_power("10", "3")[0] == (1000.0, False)
        assert _combine_power("10", "-2")[0] == (pytest.approx(0.01), False)
        assert _combine_power("10", "0")[0] == (1.0, False)
        # The run-together reading is kept, as the less likely one.
        assert (103.0, True) in _combine_power("10", "3")
        # A raised minus sign is easily lost, so the reciprocal is offered too.
        assert (pytest.approx(0.001), True) in _combine_power("10", "3")
        # Stray marks beside the base do not hide it.
        assert _combine_power("10-", "1")[0] == (10.0, False)
        # Not legibly a power of ten: nothing about the label is to be relied on.
        assert _combine_power("25", "3") == [(25.0, True), (253.0, True)]
        assert _combine_power("107", "") == [(107.0, True)]


def label(pos, *vals, anchored=True, doubtful=None, text=None):
    lab = TickLabel(pos=float(pos), box=(0, int(pos) - 5, 20, int(pos) + 5))
    lab.values = [float(v) for v in vals]
    lab.doubtful = list(doubtful) if doubtful else [False] * len(vals)
    lab.text = text if text is not None else (f"{vals[0]:g}" if vals else "")
    lab.readings = [lab.text]
    if anchored:
        lab.anchor = float(pos)
    return lab


class TestFitAxis:
    def test_clean_linear(self):
        labs = [label(100 + 50 * i, 300 - 100 * i) for i in range(4)]  # 300 at row 100 ... 0 at row 250
        fit = fit_axis(labs, increasing=False)
        assert fit.scale == "linear"
        assert len(fit.inliers) == 4
        assert fit.confidence == "high"
        assert float(fit.value_at(100)) == pytest.approx(300)
        assert float(fit.value_at(250)) == pytest.approx(0)

    def test_misread_labels_are_outvoted(self):
        # 250 read as 280 and 150 as 160: two wrong labels that happen to be
        # collinear with 100 must not beat the three right ones.
        labs = [
            label(20, 300),
            label(43, 280),
            label(66, 200),
            label(89, 160),
            label(112, 100),
            label(135, 6),
            label(158, 6),
        ]
        fit = fit_axis(labs, increasing=False)
        assert float(fit.value_at(20)) == pytest.approx(300, abs=2)
        assert float(fit.value_at(158)) == pytest.approx(0, abs=2)

    def test_alternative_readings_are_used(self):
        # Each label has the right value among its readings, but not always first.
        labs = [
            label(20, 300),
            label(43, 280, 250),
            label(66, 200),
            label(89, 180, 150),
            label(112, 100),
            label(135, 50),
            label(158, 6, 0),
        ]
        fit = fit_axis(labs, increasing=False)
        assert len(fit.inliers) == 7
        assert float(fit.value_at(158)) == pytest.approx(0, abs=1)

    def test_off_lattice_reading_rejected(self):
        # "8" where 0 belongs lies close enough to tilt a least-squares line,
        # but it is not a multiple of the tick step (50).
        labs = [label(17, 300), label(40, 250), label(63, 200), label(108, 100, anchored=False), label(154, 8)]
        fit = fit_axis(labs, increasing=False)
        assert all(v != 8 for _, v in fit.inliers)
        assert float(fit.value_at(154)) == pytest.approx(0, abs=2.5)

    def test_log_axis(self):
        labs = [label(40 + 100 * i, 10 ** (3 - i)) for i in range(4)]
        fit = fit_axis(labs, increasing=False)
        assert fit.scale == "log"
        assert float(fit.value_at(40)) == pytest.approx(1000)
        assert float(fit.value_at(190)) == pytest.approx(10 ** 1.5)

    def test_power_reading_preferred_over_run_together(self):
        # 10^3, 10^2, 10^1, 10^0 are also readable as 103, 102, 101, 100 -
        # a perfectly good linear axis.  The exponent reading must win.
        labs = [label(40 + 100 * i, 10 ** (3 - i), 103 - i, doubtful=[False, True]) for i in range(4)]
        fit = fit_axis(labs, increasing=False)
        assert fit.scale == "log"

    def test_linear_only(self):
        labs = [label(40 + 100 * i, 10 ** (3 - i)) for i in range(4)]
        fit = fit_axis(labs, increasing=False, scales=("linear",))
        assert fit is None or fit.scale == "linear"

    def test_direction_constraint(self):
        labs = [label(100 + 50 * i, 100 * i) for i in range(4)]  # values increase down the image
        assert fit_axis(labs, increasing=False) is None
        assert fit_axis(labs, increasing=True) is not None

    def test_two_labels_low_confidence(self):
        fit = fit_axis([label(100, 10), label(200, 0)], increasing=False)
        assert fit is not None and fit.confidence == "low"

    def test_too_few_labels(self):
        assert fit_axis([label(100, 10)]) is None
        assert fit_axis([]) is None

    def test_unanchored_labels_are_not_high_confidence(self):
        labs = [label(100 + 50 * i, 300 - 100 * i, anchored=False) for i in range(4)]
        assert fit_axis(labs, increasing=False).confidence == "medium"


    def test_near_misses_do_not_count(self):
        # 200, 103 and 9 lie within a pixel or two of a line, but tick values step by exactly
        # the same amount and these do not (97, then 94).
        labs = [label(100, 9, 90), label(150, 200, 209), label(200, 103, 208), label(250, 9)]
        fit = fit_axis(labs, increasing=False)
        assert len(fit.inliers) == 2
        assert fit.confidence == "low" and not fit.usable

    def test_accidental_agreement_has_an_odd_step(self):
        # Misreadings of 0, -50, -100, -150, -200: three happen to be evenly stepped (9, -100, -209),
        # but nobody draws an axis in steps of 54.5.
        labs = [label(100, 9), label(150, 3), label(200, -100), label(250, 7), label(300, -209)]
        fit = fit_axis(labs, increasing=False)
        assert len(fit.inliers) == 3 and not fit.usual_step
        assert fit.confidence == "low" and not fit.usable

    @pytest.mark.parametrize("step", [1, 2, 2.5, 5, 10, 12, 24, 0.5, 0.25, 7, 15, 45, 90, 6, 30, 1000, 0.01])
    def test_usual_steps(self, step):
        labs = [label(100 + 40 * i, step * (4 - i)) for i in range(5)]
        assert fit_axis(labs, increasing=False).usual_step

    def test_half_the_labels_on_a_usual_step_is_medium(self):
        labs = [label(100, 300), label(150, 7), label(200, 200), label(250, 3), label(300, 100), label(350, 4)]
        fit = fit_axis(labs, increasing=False)
        assert [v for _, v in fit.inliers] == [300, 200, 100]
        assert fit.confidence == "medium" and fit.usable

    def test_unread_labels_count_against_the_fit(self):
        # Seven labels were found beside the axis; only three could be read.
        labs = [label(100 + 40 * i, *(v,)) for i, v in zip((0, 2, 4), (300, 200, 100))]
        labs += [label(100 + 40 * i) for i in (1, 3, 5, 6)]
        fit = fit_axis(labs, increasing=False)
        assert fit.n_total == 7 and len(fit.inliers) == 3
        assert fit.confidence == "low" and not fit.usable
        # ... and they keep their place among the ticks: the step is 50 a tick, not 100.
        assert float(fit.value_at(340)) == pytest.approx(0)

    def test_sparse_axis_is_usable_at_low_confidence(self):
        fit = fit_axis([label(100, 300), label(300, 0)], increasing=False)
        assert fit.confidence == "low" and fit.usable

    def test_small_print_is_never_high_confidence(self):
        fit = fit_axis([label(100 + 50 * i, 300 - 100 * i) for i in range(4)], increasing=False)
        assert fit.confidence == "high"
        fit.glyph_px = 6.0
        assert fit.small_print and fit.confidence == "medium"

    def test_log_axis_needs_nearly_every_label(self):
        # Exponents are small whole numbers; three of five in step is not enough.
        labs = [label(40, 1000), label(140, 5), label(240, 10), label(340, 30), label(440, 0.1)]
        fit = fit_axis(labs, increasing=False, scales=("log",))
        assert fit.scale == "log" and len(fit.inliers) == 3
        assert fit.confidence == "low"

    def test_doubtful_readings_alone_are_not_believed(self):
        labs = [label(40 + 100 * i, 103 - i, doubtful=[True]) for i in range(4)]
        assert fit_axis(labs, increasing=False).confidence == "low"

    def test_unevenly_spaced_labels(self):
        # Hand-picked tick positions: only a line can be fitted, so nearly all labels must agree.
        labs = [label(100, 100), label(150, 50), label(190, 10), label(199, 1)]
        fit = fit_axis(labs, increasing=False, scales=("linear",))
        assert not fit.exact and len(fit.inliers) == 4
        assert fit.confidence == "medium"
        assert float(fit.value_at(200)) == pytest.approx(0, abs=0.5)


class TestRungs:
    def test_equally_spaced(self):
        assert _rungs([20.0, 43.0, 66.2, 89.0], [1.5] * 4) == [0, 1, 2, 3]

    def test_gap_leaves_a_rung_empty(self):
        assert _rungs([20.0, 43.0, 89.0, 112.0], [1.5] * 4) == [0, 1, 3, 4]

    def test_order_does_not_matter(self):
        assert _rungs([89.0, 20.0, 43.0], [1.5] * 3) == [3, 0, 1]

    def test_uneven(self):
        assert _rungs([20.0, 43.0, 80.0, 89.0], [1.5] * 4) is None

    def test_two_labels_on_one_tick(self):
        assert _rungs([20.0, 21.0, 60.0], [1.5] * 3) is None


def printed(pos, width, *readings, height=7):
    """A label found beside a y axis: *width* pixels of print, with what OCR made of it."""
    top = int(pos) - height // 2
    lab = TickLabel(pos=float(pos), box=(60 - width, top, 59, top + height - 1), anchor=float(pos))
    lab.readings = list(readings)
    return lab


class TestReadingsAgainstWidth:
    """Digits are equally wide (5 pixels here), so the width of a label gives its length."""

    def test_dropped_digit(self):
        labs = [printed(20, 19, "000", "5000"), printed(60, 19, "4000"), printed(100, 19, "3000"),
                printed(140, 19, "2000"), printed(180, 19, "1000"), printed(220, 4, "0")]
        assert _readings_that_fit(labs)[0] == ["5000"]

    def test_invented_digit(self):
        labs = [printed(20, 14, "300"), printed(60, 14, "200", "2000"), printed(100, 14, "100"), printed(140, 4, "0")]
        assert _readings_that_fit(labs)[1] == ["200"]

    def test_reading_too_short_for_any_typeface(self):
        # 12 pixels of 7-pixel print is at least two characters, whatever the others say.
        assert _readings_that_fit([printed(20, 12, "6", "0.6")]) == [["0.6"]]

    def test_fitting_readings_are_all_kept(self):
        labs = [
            printed(20, 14, "300"),
            printed(60, 14, "200", "280"),
            printed(100, 14, "100"),
            printed(140, 4, "0", "6"),
        ]
        assert _readings_that_fit(labs) == [["300"], ["200", "280"], ["100"], ["0", "6"]]

    def test_minus_sign_is_not_voted_out(self):
        # The sign was read only once in each label; the width cannot settle it, so both readings stay
        # and the direction of the axis decides.
        labs = [printed(100, 4, "0"), printed(150, 14, "25", "25", "25", "-25"), printed(200, 14, "50", "50", "-50")]
        _interpret(labs)
        assert -25.0 in labs[1].values and -50.0 in labs[2].values
        fit = fit_axis(labs, increasing=False)
        assert [v for _, v in fit.inliers] == [0.0, -25.0, -50.0]


class TestInterpret:
    def test_point_read_once_shows_up_its_loss(self):
        lab = printed(20, 17, "390", "39.0")
        _interpret([lab])
        assert dict(zip(lab.values, lab.doubtful)) == {390.0: True, 39.0: False}

    def test_axis_written_with_decimals_throughout(self):
        # "5.0" shows that every label has one decimal, so "75" and "25" have lost their points.
        labs = [printed(20, 17, "10.0"), printed(60, 12, "75"), printed(100, 12, "5.0"), printed(140, 12, "25"),
                printed(180, 12, "0.0")]
        _interpret(labs)
        assert dict(zip(labs[1].values, labs[1].doubtful)) == {75.0: True, 7.5: False}
        fit = fit_axis(labs, increasing=False)
        assert float(fit.value_at(20)) == pytest.approx(10) and len(fit.inliers) == 5

    def test_axis_with_decimals_only_where_needed(self):
        # 0, 2.5, 5, 7.5, 10: a bare 5 is a five.
        labs = [printed(20, 9, "10"), printed(60, 12, "7.5"), printed(100, 4, "5"), printed(140, 12, "2.5"),
                printed(180, 4, "0")]
        _interpret(labs)
        assert dict(zip(labs[2].values, labs[2].doubtful)) == {5.0: False, 0.5: True}
        fit = fit_axis(labs, increasing=False)
        assert float(fit.value_at(20)) == pytest.approx(10) and len(fit.inliers) == 5

    def test_leading_zero_reveals_lost_points(self):
        # 2.0, 1.5, 1.0, 0.5, 0.0 with every point lost.
        labs = [printed(20 + 40 * i, 12, t) for i, t in enumerate(["20", "15", "10", "05", "00"])]
        _interpret(labs)
        fit = fit_axis(labs, increasing=False)
        assert float(fit.value_at(20)) == pytest.approx(2.0) and float(fit.value_at(180)) == pytest.approx(0.0)
        assert len(fit.inliers) == 5

    def test_two_zeros_as_wide_as_the_other_labels_reveal_lost_points(self):
        # 3.0, 2.0, 1.0, 0.0 with every point lost.  Only "00" hints at the decimals: a lone 0 would be
        # one character wide, and this label is as wide as the rest.
        labs = [printed(20 + 40 * i, 12, t) for i, t in enumerate(["30", "20", "10", "00"])]
        _interpret(labs)
        fit = fit_axis(labs, increasing=False)
        assert [v for _, v in fit.inliers] == [3.0, 2.0, 1.0, 0.0]

    def test_zeros_misread_from_a_longer_label_are_no_evidence(self):
        # "00" for 300: too short for the label, so it is set aside and nothing is scaled down.
        labs = [printed(20, 14, "00", "300"), printed(60, 14, "200"), printed(100, 14, "100"), printed(140, 4, "0")]
        _interpret(labs)
        assert labs[1].values == [200.0] and labs[0].values == [300.0]

    def test_stray_signs_do_not_crash(self):
        labs = [printed(20, 19, "--3000"), printed(60, 19, "2000"), printed(100, 9, "05")]
        _interpret(labs)  # used to raise ValueError
        assert labs[1].values


@needs_tesseract
class TestCalibrateFromImage:
    """End to end on rendered figures: locate, read and fit."""

    @pytest.mark.parametrize(
        "case",
        [
            "A1_box_grid_blue",  # 0..100
            "A2_box_grid_red_0_300",
            "A3_box_grid_decimal_0_1",  # 0.0, 0.2 ... 1.0
            "A4_box_grid_negative",  # -50..50
            "A5_box_grid_thousands",
            "B2_L_axes_ticks_out",  # tick marks, no gridlines
            "D_dpi60",
            "D_dpi300",
            "E_jpeg_q30",
            "I1_log_y",  # 10^0 .. 10^3 written with exponents
        ],
    )
    def test_y_axis(self, render, case):
        fig = render(case)
        rgb = load_image(fig.path)
        panel = find_plot_panels(rgb)[0][0]
        fit, labels = calibrate_axis(rgb, darkness_map(rgb), panel, "y")
        assert fit is not None, [lab.readings for lab in labels]
        # Labels under 8 pixels high are read, but never with full confidence - and how many of them
        # Tesseract makes out depends on how the text was rasterised, so the fit may also be set aside.
        assert fit.small_print == (case == "D_dpi60")
        if fit.small_print:
            assert fit.confidence != "high"
            if not fit.usable:
                return
        else:
            assert fit.confidence == "high"
        lo, hi = fig.panel["ylim"]
        span = (hi - lo) if fig.panel["yscale"] == "linear" else None
        top, bottom = float(fit.value_at(fig.panel["top"])), float(fit.value_at(fig.panel["bottom"]))
        if span is None:
            assert fit.scale == "log"
            assert top == pytest.approx(hi, rel=0.02) and bottom == pytest.approx(lo, rel=0.02)
        else:
            assert fit.scale == "linear"
            assert abs(top - hi) <= 0.01 * span and abs(bottom - lo) <= 0.01 * span

    def test_end_labels_wider_than_the_axis_overhang(self, render):
        """The first and last x labels are centred on the ends of the axis and stick out past them."""
        from trace_digitiser.validation import FigureSpec

        fig = render(FigureSpec("years", xlim=(1990, 2020), xticks=(1990, 2000, 2010, 2020)))
        rgb = load_image(fig.path)
        panel = find_plot_panels(rgb)[0][0]
        fit, labels = calibrate_axis(rgb, darkness_map(rgb), panel, "x")
        assert [lab.text for lab in labels] == ["1990", "2000", "2010", "2020"]  # not 990 ... 202
        assert fit.confidence == "high"
        assert float(fit.value_at(fig.panel["left"])) == pytest.approx(1990, abs=0.3)
        assert float(fit.value_at(fig.panel["right"])) == pytest.approx(2020, abs=0.3)

    def test_x_axis(self, render):
        fig = render("A1_box_grid_blue")
        rgb = load_image(fig.path)
        panel = find_plot_panels(rgb)[0][0]
        fit, _ = calibrate_axis(rgb, darkness_map(rgb), panel, "x")
        assert fit is not None and fit.confidence == "high"
        assert float(fit.value_at(fig.panel["left"])) == pytest.approx(0, abs=0.05)
        assert float(fit.value_at(fig.panel["right"])) == pytest.approx(10, abs=0.05)

    def test_word_labels_are_not_numbers(self, vertical_2_panel_image):
        """An x axis labelled 1N, 1D, 2N ... must not be given a numeric scale."""
        rgb = load_image(vertical_2_panel_image)
        panel = find_plot_panels(rgb)[0][0]
        fit, _ = calibrate_axis(rgb, darkness_map(rgb), panel, "x")
        assert fit is None

    def test_labels_located_beside_rotated_title(self, render):
        """The rotated y-axis title sits two pixels from the tick labels and must be kept out."""
        fig = render("K1_dense_72dpi")
        rgb = load_image(fig.path)
        panel = find_plot_panels(rgb)[0][0]
        labels = find_tick_labels(rgb, darkness_map(rgb), panel, "y")
        assert 5 <= len(labels) <= 7  # 300, 250 ... 0

    @needs_published_figure
    def test_published_figure_low_resolution_labels(self, published_figure):
        """7-pixel-high labels, several misread by any single OCR attempt: the consensus still holds."""
        rgb = load_image(published_figure)
        dark = darkness_map(rgb)
        for panel in find_plot_panels(rgb)[0]:
            fit, _ = calibrate_axis(rgb, dark, panel, "y")
            assert fit is not None and len(fit.inliers) >= 4
            assert float(fit.value_at(panel["edges"]["top"])) == pytest.approx(300, abs=3)
            assert float(fit.value_at(panel["edges"]["bottom"])) == pytest.approx(0, abs=3)
