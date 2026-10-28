"""
generate_pdf.py – Convert an HTML string to a PDF file using WeasyPrint.
"""

from pathlib import Path


def generate_pdf_from_html(html: str, output_path: Path) -> None:
    """Render *html* to a PDF file at *output_path*.

    The parent directory is created automatically if it does not exist.

    Args:
        html:        Complete HTML document string.
        output_path: Destination path for the generated PDF file.

    Raises:
        ImportError: If WeasyPrint is not installed.
        OSError:     If the output file cannot be written.
    """
    try:
        from weasyprint import HTML  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "WeasyPrint is required. Install it with: pip install weasyprint"
        ) from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    HTML(string=html).write_pdf(str(output_path))
