import re

def clean_text(text):
    """
    Remove unwanted characters and normalize
    """
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)
    return text


def fix_common_errors(text):
    """
    Fix common OCR confusions
    """
    replacements = {
        'O': '0',
        'I': '1',
        'Z': '2',
        'S': '5'
        # ⚠️ Removed B→8 (too destructive for plates like AB)
    }

    corrected = ""
    for char in text:
        corrected += replacements.get(char, char)

    return corrected


def extract_plate(text):
    """
    Extract Indian license plate using flexible patterns
    """

    patterns = [

        # XX00XX0000 (MP04AB1234)
        r'([A-Z]{2})(\d{2})([A-Z]{2})(\d{4})',

        # XX0XXX0000 (DL1CAB1123)
        r'([A-Z]{2})(\d[A-Z])([A-Z]{2})(\d{4})',

        # XX00X0000 (TN23L4547)
        r'([A-Z]{2})(\d{2})([A-Z])(\d{4})',

        # XX00XX000 (KA01AB123)
        r'([A-Z]{2})(\d{2})([A-Z]{2})(\d{3})'
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return "".join(match.groups())

    return text  # fallback


def process_plate_text(raw_text):
    """
    Full pipeline
    """
    cleaned = clean_text(raw_text)
    corrected = fix_common_errors(cleaned)
    extracted = extract_plate(corrected)

    return extracted
