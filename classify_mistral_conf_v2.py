import ollama
import pandas as pd
import json
import re
from tqdm import tqdm

def classify_texts_with_conf(texts):
    # Prompt enforcing JSON format
    prompt = f"""
You are a text classifier. For each sentence below, classify it as FACT or OPINION.
Return the result strictly as a JSON list where each element has:
- 'label': either "FACT" or "OPINION"
- 'confidence': a number from 0 to 100 indicating how confident you are.

Example:
[
  {{"label": "FACT", "confidence": 93}},
  {{"label": "OPINION", "confidence": 68}}
]

Classify these:
{texts}
"""

    response = ollama.generate(model="mistral:latest", prompt=prompt)
    raw_output = response.get("response", "").strip()

    try:
        # Try to parse JSON directly
        parsed = json.loads(raw_output)
        labels = [item.get("label", "UNKNOWN") for item in parsed]
        confs = [item.get("confidence", 50) for item in parsed]
    except json.JSONDecodeError:
        # If model didn’t output valid JSON, fallback regex
        lines = raw_output.splitlines()
        labels, confs = [], []
        for line in lines:
            m = re.search(r"(FACT|OPINION)", line, re.I)
            c = re.search(r"(\d{1,3})", line)
            labels.append(m.group(1).upper() if m else "OPINION")
            confs.append(int(c.group(1)) if c else 50)

    return labels, confs


def main(csv_path, text_col, out_path, batch_size=20, truncate_chars=1000):
    df = pd.read_csv(csv_path)
    results_labels = []
    results_confs = []

    for i in tqdm(range(0, len(df), batch_size)):
        batch_texts = df[text_col].iloc[i:i+batch_size].fillna("").astype(str)
        joined_texts = "\n".join(
            f"{j+1}. {txt[:truncate_chars]}" for j, txt in enumerate(batch_texts)
        )
        labels, confs = classify_texts_with_conf(joined_texts)
        results_labels.extend(labels)
        results_confs.extend(confs)

    df["label_mistral"] = results_labels[:len(df)]
    df["conf_mistral"] = results_confs[:len(df)]
    df.to_csv(out_path, index=False)
    print(f"✅ Saved predictions with confidence to {out_path}")
    print(f"📊 Average confidence: {sum(results_confs)/len(results_confs):.2f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--text-col", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--truncate-chars", type=int, default=1000)
    args = parser.parse_args()

    main(args.csv, args.text_col, args.out, args.batch_size, args.truncate_chars)
