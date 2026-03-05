import pandas as pd

INPUT_XLSX = "FSTE100-LSEG.xlsx"   # file location (repo root). change if needed
OUTPUT_PY = "src/tickers_ftse100.py"

CODE_COL = "Code"
NAME_COL = "Name"   # <-- change if your column is called something else


def lseg_to_yahoo(code: str) -> str:
    c = str(code).strip()
    c = c.rstrip(".")          # e.g. "RR." -> "RR"
    c = c.replace(".", "-")    # e.g. "BT.A" -> "BT-A"
    return f"{c}.L"


def safe_comment(text: str) -> str:
    # Keep comments clean and avoid breaking the file
    return str(text).replace("\n", " ").replace("\r", " ").strip()


def main():
    df = pd.read_excel(INPUT_XLSX)

    if CODE_COL not in df.columns:
        raise ValueError(f"Expected column '{CODE_COL}'. Found: {list(df.columns)}")
    if NAME_COL not in df.columns:
        raise ValueError(f"Expected column '{NAME_COL}'. Found: {list(df.columns)}")

    rows = df[[CODE_COL, NAME_COL]].dropna(subset=[CODE_COL]).copy()

    tickers = []
    for _, row in rows.iterrows():
        code = row[CODE_COL]
        name = row[NAME_COL] if pd.notna(row[NAME_COL]) else ""
        yahoo = lseg_to_yahoo(code)
        tickers.append((yahoo, safe_comment(name)))

    # remove duplicates while preserving order
    seen = set()
    tickers_unique = []
    for tkr, nm in tickers:
        if tkr not in seen:
            seen.add(tkr)
            tickers_unique.append((tkr, nm))

    print("Ticker count:", len(tickers_unique))

    with open(OUTPUT_PY, "w", encoding="utf-8") as f:
        f.write("FTSE100_TICKERS = [\n")
        for tkr, nm in tickers_unique:
            if nm:
                f.write(f'    "{tkr}",  # {nm}\n')
            else:
                f.write(f'    "{tkr}",\n')
        f.write("]\n")

    print("Saved:", OUTPUT_PY)


if __name__ == "__main__":
    main()