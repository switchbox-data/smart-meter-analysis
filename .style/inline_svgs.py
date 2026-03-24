"""Inline SVG figures into Quarto-rendered HTML files.

Replaces <img src="...svg"> tags with the raw <svg> markup so that inline
SVGs participate in the page's CSS cascade and the browser's text renderer
(with font hinting) handles chart text.

Also patches lightbox links wrapping inlined SVGs so that clicking them
opens a modal showing a clone of the inline SVG (preserving page fonts),
rather than loading the external SVG file (which would lose fonts).

Usage (from a report directory, after quarto render):

    uv run python ../.style/inline_svgs.py docs/

The script processes every .html file under the given directory.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Lightweight modal for inline SVGs.  Clones the SVG so page @font-face
# rules apply.  Supports click-to-close, Escape, and left/right navigation
# between figures on the same page.
_SVG_LIGHTBOX_SCRIPT = """
<style>
.svg-lightbox-overlay {
  position: fixed; top: 0; left: 0; width: 100%; height: 100%;
  background: rgba(0,0,0,.92); z-index: 99999;
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  cursor: pointer; box-sizing: border-box; padding: 2vh 2vw;
}
.svg-lightbox-overlay > svg {
  max-width: 96vw; max-height: calc(96vh - 4em);
  height: auto; width: auto;
}
.svg-lightbox-caption {
  display: flex; align-items: baseline; flex-wrap: wrap;
  background: #fff; border-radius: 4px;
  padding: .5em .9em; margin-top: .7em;
  font-size: 1rem; line-height: 1.5; max-width: 80vw;
}
.svg-lightbox-caption .fig-num {
  font-family: "GT-Planar-Black", "GT-Planar", "GT Planar", sans-serif;
  color: #fc9706; margin-right: 5px; white-space: nowrap;
}
.svg-lightbox-caption .fig-desc {
  font-family: "GT-Planar", "GT Planar", sans-serif;
  color: #a9aaae;
}
</style>
<script>
(function() {
  var links = document.querySelectorAll('a.lightbox-svg');
  if (!links.length) return;
  var overlay, current, ordered = Array.from(links);

  function show(a) {
    var svg = a.querySelector('svg');
    if (!svg) return;
    if (overlay) overlay.remove();
    current = a;

    overlay = document.createElement('div');
    overlay.className = 'svg-lightbox-overlay';

    var clone = svg.cloneNode(true);
    clone.removeAttribute('class');
    clone.removeAttribute('width');
    var vb = clone.getAttribute('viewBox');
    if (vb) {
      var parts = vb.split(/[\\s,]+/);
      var w = parseFloat(parts[2]), h = parseFloat(parts[3]);
      if (w && h) clone.style.aspectRatio = w + '/' + h;
    }
    overlay.appendChild(clone);

    var title = a.getAttribute('title');
    if (title) {
      var cap = document.createElement('div');
      cap.className = 'svg-lightbox-caption';
      var colonIdx = title.indexOf(':');
      if (colonIdx > 0 && colonIdx < 20) {
        cap.innerHTML = '<span class="fig-num">' +
          title.substring(0, colonIdx + 1) + '</span>' +
          '<span class="fig-desc">' + title.substring(colonIdx + 1) + '</span>';
      } else {
        cap.innerHTML = '<span class="fig-desc">' + title + '</span>';
      }
      overlay.appendChild(cap);
    }
    overlay.addEventListener('click', close);
    document.body.appendChild(overlay);
  }

  function close() {
    if (overlay) { overlay.remove(); overlay = null; current = null; }
  }

  function nav(dir) {
    if (!current) return;
    var i = ordered.indexOf(current) + dir;
    if (i >= 0 && i < ordered.length) show(ordered[i]);
  }

  document.addEventListener('keydown', function(e) {
    if (!overlay) return;
    if (e.key === 'Escape') close();
    else if (e.key === 'ArrowLeft') nav(-1);
    else if (e.key === 'ArrowRight') nav(1);
  });

  links.forEach(function(a) {
    a.addEventListener('click', function(e) {
      e.preventDefault();
      e.stopPropagation();
      show(a);
    });
  });
})();
</script>
"""


def _read_svg_body(svg_path: Path) -> str:
    """Read an SVG file and strip the XML declaration and DOCTYPE."""
    raw = svg_path.read_text(encoding="utf-8")
    raw = re.sub(r"<\?xml[^?]*\?>", "", raw).strip()
    raw = re.sub(r"<!DOCTYPE[^>]*>", "", raw).strip()
    return raw


def _make_svg_fixed_width(svg_markup: str, classes: str) -> str:
    """Modify the <svg> root element for fixed-width display.

    Uses the viewBox width (in pt, which maps 1:1 to CSS px) as the display
    width so that matplotlib font sizes render at the intended pixel size
    regardless of container width.  ``max-width: 100%`` prevents overflow on
    narrow viewports.  The SVG is left-aligned in the column (no centering).
    """
    vb_match = re.search(r'viewBox="([^"]*)"', svg_markup)
    if vb_match:
        parts = vb_match.group(1).split()
        vb_width = parts[2] if len(parts) >= 3 else None
    else:
        vb_width = None

    def replace_svg_tag(m: re.Match[str]) -> str:
        tag_content = m.group(1)
        tag_content = re.sub(r'\bwidth="[^"]*"', "", tag_content)
        tag_content = re.sub(r'\bheight="[^"]*"', "", tag_content)
        tag_content = tag_content.strip()
        if vb_width:
            style = "max-width:100%;height:auto;display:block;margin:0;"
            return f'<svg {tag_content} width="{vb_width}" style="{style}" class="{classes}" role="img">'
        return f'<svg {tag_content} width="100%" class="{classes}" role="img">'

    return re.sub(r"<svg\b([^>]*)>", replace_svg_tag, svg_markup, count=1)


_IMG_SVG_RE = re.compile(
    r'<img\s+src="([^"]+\.svg)"'
    r'(?:\s+class="([^"]*)")?'
    r"[^>]*/?>",
    re.IGNORECASE,
)


def inline_svgs_in_html(html_path: Path) -> int:
    """Replace <img src="...svg"> with inline <svg> in a single HTML file.

    Also renames the wrapping lightbox class so GLightbox ignores these links,
    and injects a lightweight modal script for SVG zoom.

    Returns the number of replacements made.
    """
    html_text = html_path.read_text(encoding="utf-8")
    html_dir = html_path.parent
    count = 0

    def _replace(m: re.Match[str]) -> str:
        nonlocal count
        src = m.group(1)
        classes = m.group(2) or ""
        svg_path = html_dir / src
        if not svg_path.exists():
            return m.group(0)
        svg_body = _read_svg_body(svg_path)
        svg_body = _make_svg_fixed_width(svg_body, classes)
        count += 1
        return svg_body

    new_html = _IMG_SVG_RE.sub(_replace, html_text)

    if count > 0:
        # Rename lightbox class only on <a> tags that now contain an inline
        # <svg> (i.e. formerly had an <img ...svg> that was replaced).
        # GLightbox uses selector ".lightbox" — renaming to "lightbox-svg"
        # keeps GLightbox from intercepting these clicks while leaving
        # lightbox links wrapping <img ...png> untouched.
        new_html = re.sub(
            r'(class="lightbox )([^"]*"[^>]*>)\s*(<svg\b)',
            r'class="lightbox-svg \2\3',
            new_html,
        )

        # Inject the lightweight SVG modal script before </body>
        new_html = new_html.replace("</body>", _SVG_LIGHTBOX_SCRIPT + "</body>")

        html_path.write_text(new_html, encoding="utf-8")
    return count


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <docs-directory>", file=sys.stderr)
        sys.exit(1)

    docs_dir = Path(sys.argv[1])
    if not docs_dir.is_dir():
        print(f"Error: {docs_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    total = 0
    for html_file in sorted(docs_dir.rglob("*.html")):
        n = inline_svgs_in_html(html_file)
        if n > 0:
            print(f"  {html_file.relative_to(docs_dir)}: inlined {n} SVG(s)")
            total += n

    # Remove external SVG figure files — they're now redundant since the
    # SVGs are inlined in the HTML and the lightbox modal clones from there.
    removed = 0
    for svg_file in sorted(docs_dir.rglob("figure-html/*.svg")):
        svg_file.unlink()
        removed += 1

    print(f"Inlined {total} SVG(s), removed {removed} external SVG(s) in {docs_dir}")


if __name__ == "__main__":
    main()
