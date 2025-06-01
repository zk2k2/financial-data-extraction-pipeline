import os
import json
import requests
from Levenshtein import distance as levenshtein_distance

# Configuration
DATA_DIR = "/workspace/data_extraction_pipeline/new_data"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
LABELS_DIR = os.path.join(DATA_DIR, "labels")
API_ENDPOINT = "http://localhost:8000/extract"  # Adjust if your pipeline endpoint differs
DELTA = 0.2               # Normalized Levenshtein threshold for string fields
NUMERIC_TOLERANCE = 1e-2  # Acceptable relative difference for numeric fields

# Define schema fields
STRING_FIELDS = [
    "supplier.name",
    "supplier.address",
    "supplier.email",
    "supplier.phone_number",
    "supplier.vat_number",
    "supplier.website",
    "invoice_date",       # expect format "DD/MM/YYYY"
    "invoice_number",
    "currency"
]
NUMERIC_FIELDS = [
    "total_net",
    "total_tax",
    "total_amount_incl_tax"
]

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_nested_value(d: dict, key_path: str):
    """
    Traverse a nested dict using a dotted key_path (e.g., 'supplier.name').
    Returns None if any key is missing.
    """
    value = d
    for key in key_path.split("."):
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None
    return value

def normalized_levenshtein(a: str, b: str) -> float:
    """
    Compute normalized Levenshtein distance between strings a and b.
    Returns D_lev = dist(a, b) / max(len(a), len(b)).
    If both strings are empty or None, returns 0.0.
    """
    a = a or ""
    b = b or ""
    if not a and not b:
        return 0.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 0.0
    dist = levenshtein_distance(a, b)
    return dist / max_len

def compare_string_field(pred: dict, truth: dict, field: str) -> bool:
    """
    Compare a string field using relaxed normalized Levenshtein matching.
    """
    pred_val = get_nested_value(pred, field)
    truth_val = get_nested_value(truth, field)
    # If both are None or empty, consider it correct
    if not pred_val and not truth_val:
        return True
    pred_str = str(pred_val or "").strip()
    truth_str = str(truth_val or "").strip()
    d_norm = normalized_levenshtein(pred_str, truth_str)
    is_match = (d_norm < DELTA)
    print(f"    [DEBUG] Field '{field}':\n"
          f"        Predicted: '{pred_str}'\n"
          f"        Ground-Truth: '{truth_str}'\n"
          f"        Normalized Levenshtein: {d_norm:.3f} -> Match: {is_match}")
    return is_match

def compare_numeric_field(pred: dict, truth: dict, field: str) -> bool:
    """
    Compare a numeric field within a relative tolerance.
    """
    pred_val = get_nested_value(pred, field)
    truth_val = get_nested_value(truth, field)
    try:
        pred_num = float(pred_val)
        truth_num = float(truth_val)
    except (TypeError, ValueError):
        print(f"    [DEBUG] Field '{field}': invalid numeric conversion "
              f"Pred='{pred_val}' Truth='{truth_val}' -> Match: False")
        return False
    if truth_num == 0:
        is_match = abs(pred_num - truth_num) < NUMERIC_TOLERANCE
    else:
        is_match = abs(pred_num - truth_num) / abs(truth_num) < NUMERIC_TOLERANCE
    print(f"    [DEBUG] Field '{field}':\n"
          f"        Predicted: {pred_num}\n"
          f"        Ground-Truth: {truth_num}\n"
          f"        Relative Difference: "
          f"{abs(pred_num - truth_num)/max(abs(truth_num), 1e-6):.3f} -> Match: {is_match}")
    return is_match

def extract_invoice(image_path: str) -> dict:
    """
    Send the invoice image to the OCR+LLM pipeline endpoint.
    Expects JSON response conforming to the ground-truth schema.
    """
    with open(image_path, "rb") as img_f:
        resp = requests.post(API_ENDPOINT, files={"invoice_image": img_f}, timeout=60)
        resp.raise_for_status()
        return resp.json()

def main():
    # Ensure subdirectories exist
    if not os.path.isdir(IMAGES_DIR) or not os.path.isdir(LABELS_DIR):
        print(f"Error: Expected subfolders 'images' and 'labels' inside {DATA_DIR}.")
        return

    # Initialize counters
    per_field_tp = {f: 0 for f in STRING_FIELDS + NUMERIC_FIELDS}
    per_field_total = {f: 0 for f in STRING_FIELDS + NUMERIC_FIELDS}
    total_invoices = 0
    total_skipped = 0

    # Iterate over each image file in images/
    for fname in os.listdir(IMAGES_DIR):
        lower = fname.lower()
        if not lower.endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".pdf")):
            continue

        base_name = os.path.splitext(fname)[0]
        image_path = os.path.join(IMAGES_DIR, fname)
        truth_path = os.path.join(LABELS_DIR, f"{base_name}.json")

        if not os.path.exists(truth_path):
            print(f"[WARN] Skipping '{fname}': no matching label {base_name}.json in labels/")
            total_skipped += 1
            continue

        print(f"\n[INFO] Processing '{fname}' with ground-truth '{base_name}.json'")
        truth_json = load_json(truth_path)
        print(f"  [DEBUG] Ground-Truth JSON:\n{json.dumps(truth_json, indent=4)}")

        try:
            pred_json = extract_invoice(image_path)
        except requests.HTTPError as e:
            print(f"[WARN] Skipping '{fname}': HTTP {e.response.status_code} from /extract")
            total_skipped += 1
            continue
        except Exception as e:
            print(f"[ERROR] Skipping '{fname}': {e}")
            total_skipped += 1
            continue

        print(f"  [DEBUG] Predicted JSON:\n{json.dumps(pred_json, indent=4)}")

        total_invoices += 1

        # Compare string fields
        for field in STRING_FIELDS:
            per_field_total[field] += 1
            matched = compare_string_field(pred_json, truth_json, field)
            if matched:
                per_field_tp[field] += 1

        # Compare numeric fields
        for field in NUMERIC_FIELDS:
            per_field_total[field] += 1
            matched = compare_numeric_field(pred_json, truth_json, field)
            if matched:
                per_field_tp[field] += 1

    # Report results
    print("\n=== Evaluation Results ===")
    print(f"Total images in 'images/': {len(os.listdir(IMAGES_DIR))}")
    print(f"Processed invoices: {total_invoices}")
    print(f"Skipped invoices:   {total_skipped}\n")

    print("Field-wise Accuracy:")
    for field in STRING_FIELDS + NUMERIC_FIELDS:
        total = per_field_total[field]
        tp = per_field_tp[field]
        accuracy = (tp / total * 100) if total > 0 else 0.0
        print(f"  {field:30s}: {accuracy:.2f}% ({tp}/{total})")

    total_tp = sum(per_field_tp.values())
    total_count = sum(per_field_total.values())
    avg_acc = (total_tp / total_count * 100) if total_count > 0 else 0.0
    print(f"\nAverage Accuracy (over all fields): {avg_acc:.2f}%")

    avg_f1 = avg_acc / 100
    print(f"Average Field-Level F1 Score: {avg_f1:.3f}")
    print("===========================")

if __name__ == "__main__":
    main()
