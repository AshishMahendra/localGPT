import pdfplumber
from .pdf_processing import preprocess_text
import re


def find_text_positions(page, search_text):
    """Improved logic to handle text matches that start or end mid-word."""
    search_text = preprocess_text(search_text)
    words = page.extract_words(keep_blank_chars=True)
    page_text = preprocess_text(" ".join([word["text"] for word in words]))

    word_positions = []
    current_index = 0
    for word in words:
        start = current_index
        end = start + len(preprocess_text(word["text"]))
        word_positions.append((start, end, word))
        current_index = end + 1  # Account for space between words

    pattern = re.compile(re.escape(search_text), re.IGNORECASE)
    matches = list(pattern.finditer(page_text))

    bounding_boxes = []
    for match in matches:
        match_start, match_end = match.start(), match.end()
        for start, end, word in word_positions:
            if start < match_end and end > match_start:  # Check for any overlap
                bounding_boxes.append((word["x0"], word["top"], word["x1"], word["bottom"]))

    return bounding_boxes


def highlight_text_in_pdf(pdf_path, page_number, highlight_text):
    """Highlight the specified text on the given page with enhanced debugging."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_number - 1]
            words = page.extract_words(keep_blank_chars=True)
            bounding_boxes = find_text_positions(page, highlight_text)

            if not bounding_boxes:
                print("No matching text found.")
                return None

            page_image = page.to_image(resolution=400)
            for box in bounding_boxes:
                page_image.draw_rect(box, fill=(255, 255, 0, 64), stroke="orange", stroke_width=3)

            output_file_path = f"highlighted_page_{page_number}.png"
            page_image.save(output_file_path, quality=95)
            print(f"Highlighted text saved to {output_file_path}")

            return output_file_path

    except Exception as e:
        print(f"An error occurred: {e}")
        return str(e)
