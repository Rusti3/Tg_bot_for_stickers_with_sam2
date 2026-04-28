from __future__ import annotations

from sticker_bot.bot.handlers import parse_add_command_args


def test_parse_add_command_defaults():
    options = parse_add_command_args(None, 10)
    assert options.w_count == 1
    assert options.back_mode == "none"
    assert options.tolerance == 10


def test_parse_add_command_accepts_grid_and_back_modes():
    options = parse_add_command_args("5 back=auto", 10)
    assert options.w_count == 5
    assert options.back_mode == "auto"

    color_options = parse_add_command_args("15 back=#ffffff20", 10)
    assert color_options.w_count == 10
    assert color_options.back_mode == "#ffffff20"
    assert color_options.tolerance == 20

