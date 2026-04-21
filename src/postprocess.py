import re

# All valid Indian state / UT codes
VALID_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN",
    "GA", "GJ", "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD",
    "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "PB", "PY", "RJ",
    "SK", "TN", "TR", "TS", "UK", "UP", "WB",
}

# OCR confusions: digit ↔ letter
DIGIT_TO_LETTER = {
    "0": "O", "1": "I", "2": "Z", "5": "S",
    "8": "B", "6": "G", "4": "A",
}

LETTER_TO_DIGIT = {v: k for k, v in DIGIT_TO_LETTER.items()}


def _to_letter(ch):
    """Force a character to be a letter (fix digit→letter confusions)."""
    return DIGIT_TO_LETTER.get(ch, ch)


def _to_digit(ch):
    """Force a character to be a digit (fix letter→digit confusions)."""
    return LETTER_TO_DIGIT.get(ch, ch)


def clean_text(text):
    """Remove unwanted characters and normalise."""
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)
    return text


def _positional_correct(text, mask):
    """
    Apply position-aware correction.
    mask uses: 'L' = letter, 'D' = digit, '?' = keep as-is.
    """
    out = []
    for ch, m in zip(text, mask):
        if m == "L":
            out.append(_to_letter(ch))
        elif m == "D":
            out.append(_to_digit(ch))
        else:
            out.append(ch)
    return "".join(out)


def _is_valid_plate(corrected, mask):
    """
    Check that every position matches its expected type
    AND that the district code (first 1-2 digits after state) is 01-99.
    """
    for ch, m in zip(corrected, mask):
        if m == "L" and not ch.isalpha():
            return False
        if m == "D" and not ch.isdigit():
            return False

    # Validate district code is 01-99
    district_str = "".join(
        ch for ch, m in zip(corrected, mask[:2]) if m == "D"
    )
    # For DL-style (mask "DLLL..."): district is a single digit, must be 1-9
    # For standard (mask "DDLL..."): district is two digits, must be 01-99
    if district_str and int(district_str) == 0:
        return False

    return True


def _match_state(text):
    """
    Try to identify a valid state code at the start of text.
    Returns (state_code, remaining_text) or (None, text).
    """
    if len(text) < 2:
        return None, text

    # Try correcting first two chars as letters
    corrected = _to_letter(text[0]) + _to_letter(text[1])
    if corrected in VALID_STATE_CODES:
        return corrected, text[2:]

    # Raw prefix already valid?
    if text[:2] in VALID_STATE_CODES:
        return text[:2], text[2:]

    return None, text


# Indian plate formats AFTER the 2-letter state code.
# Listed most-specific-first.
#
#   Format A:  DD LL DDDD   e.g. MP 04 AB 1234   (8 chars)
#   Format B:  DL LL DDDD   e.g. DL 1C AB 1123   (8 chars)
#   Format C:  DD L  DDDD   e.g. TN 23 L  4547   (7 chars)
#   Format D:  DD LL DDD                           (7 chars)
#
PLATE_FORMATS = [
    ("DDLLDDDD", r'\d{2}[A-Z]{2}\d{4}'),   # A
    ("DLLLDDDD", r'\d[A-Z]{3}\d{4}'),       # B
    ("DDLDDDD",  r'\d{2}[A-Z]\d{4}'),       # C
    ("DDLLDDD",  r'\d{2}[A-Z]{2}\d{3}'),    # D
]


def extract_plate(text):
    """
    Extract an Indian license plate using positional correction.
    Tries each known format, picks the first valid match.
    Falls back to cleaned text if nothing matches.
    """

    state, rest = _match_state(text)
    if state is None:
        return text  # can't identify state → return as-is

    for mask, regex in PLATE_FORMATS:
        need = len(mask)
        if len(rest) < need:
            continue

        candidate = rest[:need]
        corrected = _positional_correct(candidate, mask)

        if re.fullmatch(regex, corrected) and _is_valid_plate(corrected, mask):
            return state + corrected

    # No format matched — return state + raw remainder
    return state + rest


def is_valid_indian_plate(text):
    """
    Final validation: does text look like a valid Indian plate?
    Must start with a known state code and match one of the formats.
    """
    if len(text) < 7:
        return False

    state = text[:2]
    if state not in VALID_STATE_CODES:
        return False

    rest = text[2:]
    for mask, regex in PLATE_FORMATS:
        if len(rest) == len(mask) and re.fullmatch(regex, rest):
            # Also check district != 00
            district_str = "".join(
                ch for ch, m in zip(rest, mask[:2]) if m == "D"
            )
            if district_str and int(district_str) > 0:
                return True

    return False


def process_plate_text(raw_text):
    """Full postprocessing pipeline."""
    cleaned = clean_text(raw_text)
    extracted = extract_plate(cleaned)

    if is_valid_indian_plate(extracted):
        return extracted

    return "NOT LEGIBLE"
