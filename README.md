# Quiet Galaxy art tools

The scripts behind the [Quiet Galaxy](https://github.com/avinashjoshi/omarchy-quiet-galaxy-theme)
Omarchy theme, kept here because a theme repository should hold only the theme.

- `cvd.py`: simulates protanopia (Machado, Oliveira & Fernandes 2009, full
  severity) and measures the theme palette's contrast and colour separation;
  `--search` looks for a better set of hues.
- `generate.py`: draws the hairline wallpapers, their intro animations, the two
  looping video wallpapers, the boot-splash emblem and the contact sheets.
- `prompts/`: the prompts used for the AI-generated wallpapers.

Both scripts read and write a checkout of the theme. By default that is
`../omarchy-quiet-galaxy-theme`; point `THEME_DIR` somewhere else to override.

```bash
uv run --with numpy cvd.py                                   # palette report
uv run --with numpy cvd.py --search                          # search for better hues
uv run --with numpy --with pillow generate.py stills         # line-art wallpapers
uv run --with numpy --with pillow generate.py videos         # intros and loops (needs ffmpeg, ~10 minutes)
uv run --with numpy --with pillow generate.py emblem unlock-preview contact-sheet
```

MIT, see [LICENSE](LICENSE).
