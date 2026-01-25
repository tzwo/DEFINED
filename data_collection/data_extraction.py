#!/usr/bin/env python3
import os
import re
import argparse
from pathlib import Path

def read_text(p: Path) -> str:
    with open(p, "r", encoding="utf-8") as f:
        return f.read()

def write_text(p: Path, s: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(s)

def find_index(lines, start, begin_at=0):
    for i in range(begin_at, len(lines)):
        if start in lines[i]:
            return i
    return -1

def extract_range(lines, start_kw, end_kw, begin_at=0):
    s_idx = find_index(lines, start_kw, begin_at)
    if s_idx == -1:
        return ("", len(lines))
    e_idx = find_index(lines, end_kw, s_idx + 1)
    if e_idx == -1:
        e_idx = len(lines)
    return ("\n".join(lines[s_idx:e_idx]), e_idx)

SPEAKER_RE = re.compile(r"^\s*说话人\s*([0-9]+)\s*$")

def select_monologue(text):
    lines = text.splitlines()
    current = None
    buckets = {}
    added_counts = {}

    mic_keywords = [
        "麦克风", "话筒", "开麦", "声音", "听见", "听得到", "听得见", "能听到", "能听见",
        "听清楚", "试音", "测试", "有声音吗", "没声音", "音量", "音响","听清","设备","问题"
    ]

    def is_mic_short_line(s: str) -> bool:
        t = s.strip()
        if len(t) == 0:
            return False
        if len(t) > 30:
            return False
        for kw in mic_keywords:
            if kw in t:
                return True
        # 常见问句简写
        if t.endswith("吗？") and ("听" in t or "麦" in t or "声" in t):
            return True
        return False
    for ln in lines:
        m = SPEAKER_RE.match(ln)
        if m:
            current = m.group(1)
            buckets.setdefault(current, [])
            continue
        if current is None:
            continue
        if ln.strip() == "":
            continue
        # 段首前两行若为麦克风相关短句则跳过
        added = added_counts.get(current, 0)
        if added < 2 and is_mic_short_line(ln):
            continue
        buckets.setdefault(current, []).append(ln)
        added_counts[current] = added + 1
    best_id = None
    best_len = -1
    for k, v in buckets.items():
        l = sum(len(x) for x in v)
        if l > best_len:
            best_len = l
            best_id = k
    if best_id is None:
        return ""
    return "\n".join(buckets[best_id]).strip()

def extract_sections(raw_text):
    lines = raw_text.splitlines()
    header = lines[0].strip() if lines else ""
    pos = 0
    segs = {}
    pro1_text, pos = extract_range(lines, "正方陈词一", "反方陈词一", pos)
    segs["正方一辩立论"] = select_monologue(pro1_text)
    con1_text, pos = extract_range(lines, "反方陈词一", "反方质询一", pos)
    segs["反方一辩立论"] = select_monologue(con1_text)
    pro2_text, pos = extract_range(lines, "回合二的正方陈词", "反方质询二", pos)
    segs["正方二辩陈词"] = select_monologue(pro2_text)
    con2_text, pos = extract_range(lines, "回合二的反方陈词", "正方质询二", pos)
    segs["反方二辩陈词"] = select_monologue(con2_text)
    con3_text, pos = extract_range(lines, "反方质询小结", "正方质询小结", pos)
    segs["反方三辩小结"] = select_monologue(con3_text)
    pro3_text, pos = extract_range(lines, "正方质询小结", "对辩环节", pos)
    segs["正方三辩小结"] = select_monologue(pro3_text)
    idx_con_sum = find_index(lines, "反方总结陈词", pos)
    idx_pro_sum = find_index(lines, "正方总结陈词", pos)
    last_idx = find_index(lines, "已全部结束", pos)
    if last_idx == -1:
        last_idx = len(lines)
    if idx_con_sum != -1 and idx_pro_sum != -1:
        if idx_con_sum < idx_pro_sum:
            con4_text = "\n".join(lines[idx_con_sum:idx_pro_sum])
            pro4_text = "\n".join(lines[idx_pro_sum:last_idx])
        else:
            pro4_text = "\n".join(lines[idx_pro_sum:idx_con_sum])
            con4_text = "\n".join(lines[idx_con_sum:last_idx])
    elif idx_con_sum != -1 and idx_pro_sum == -1:
        con4_text = "\n".join(lines[idx_con_sum:last_idx])
        pro4_text = ""
    elif idx_con_sum == -1 and idx_pro_sum != -1:
        pro4_text = "\n".join(lines[idx_pro_sum:last_idx])
        con4_text = ""
    else:
        con4_text = ""
        pro4_text = ""
    segs["反方四辩结辩"] = select_monologue(con4_text)
    segs["正方四辩结辩"] = select_monologue(pro4_text)
    return header, segs

def render_output(header, segs):
    parts = [header, ""]
    order = [
        "正方一辩立论",
        "反方一辩立论",
        "正方二辩陈词",
        "反方二辩陈词",
        "反方三辩小结",
        "正方三辩小结",
        "反方四辩结辩",
        "正方四辩结辩",
    ]
    for key in order:
        parts.append(f"{key}：")
        parts.append(segs.get(key, ""))
    return "\n".join(parts).strip() + "\n"

def process_file(in_path: Path, out_dir: Path) -> int:
    raw = read_text(in_path)
    header, segs = extract_sections(raw)
    out_text = render_output(header, segs)
    out_path = out_dir / in_path.name
    write_text(out_path, out_text)
    count = sum(1 for k in segs if (segs[k] or "").strip())
    return count

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="")
    ap.add_argument("--output_dir", default="")
    args = ap.parse_args()
    in_dir = Path(args.input_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    total = 0
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".txt"):
            continue
        total += process_file(in_dir / name, out_dir)
    print(total)

if __name__ == "__main__":
    main()