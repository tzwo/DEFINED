import argparse
import json
import os
import random
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from peft import PeftModelForCausalLM
from prompt.prompt_midium import CREATIVITY_PROMPT_DEBATE
from jinja2 import Environment, FileSystemLoader
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="从 debate/match.json 读取比赛数据并进行推理生成")
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="基础模型路径或HF名称",
    )
    parser.add_argument("--adapter_path", type=str, default=None, help="PEFT适配器路径(可选)")
    parser.add_argument(
        "--match_path",
        type=str,
        default="/Users/woyu/Desktop/learning_material/25_a/SII/edu/code/sotopia-rl/debate/match.json",
        help="比赛数据JSON路径",
    )
    parser.add_argument("--output_path", type=str, required=True, help="生成后的输出JSON路径")
    parser.add_argument("--max_length", type=int, default=1024, help="最大输出长度")
    parser.add_argument("--max_new_tokens", type=int, default=1536, help="最大生成新tokens数")
    parser.add_argument("--seed", type=int, default=1024, help="随机种子")
    parser.add_argument("--batch_size", type=int, default=1, help="批量生成的batch大小")
    parser.add_argument("--template_path", type=str, default=None, help="Jinja模板路径")
    parser.add_argument("--mode", type=int, default=4, help="推理模式: 1=背景+上一轮, 2=仅辩题+持方, 3=背景+上一轮+本轮参考+目标分, 4=总结input并追加")
    return parser.parse_args()


def load_model_and_tokenizer(model_path, adapter_path=None):
    print(f"Loading base model (no LoRA/QLoRA): {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Ensure pad token is set for batched generation
    if tokenizer.pad_token_id is None:
        if getattr(tokenizer, "eos_token_id", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    try:
        tokenizer.padding_side = "left"
    except Exception:
        pass
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    if adapter_path:
        ap = adapter_path
        if os.path.exists(os.path.join(ap, "adapter_model.safetensors")) or os.path.exists(os.path.join(ap, "adapter_model.bin")):
            model = PeftModelForCausalLM.from_pretrained(base_model, ap)
        else:
            model = base_model
    else:
        model = base_model
    model.eval()
    return model, tokenizer


def build_prompt(example_input: str) -> str:
    """Return the prompt to feed to the model.

    Prefer dataset 'input' if present (it usually includes the full debate prompt).
    Otherwise, fall back to CREATIVITY_PROMPT_DEBATE from debate.prompt.
    """
    if example_input:
        return example_input
    return CREATIVITY_PROMPT_DEBATE


def load_template(template_path):
    template_dir = os.path.dirname(template_path)
    template_file = os.path.basename(template_path)
    if not template_dir:
        template_dir = "."
    env = Environment(loader=FileSystemLoader(template_dir))
    env.filters['tojson'] = lambda obj: json.dumps(obj)
    return env.get_template(template_file)


def generate_response(model, tokenizer, prompt, max_length=512):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
        )
    response = tokenizer.decode(output[0], skip_special_tokens=False)
    return response


def generate_batch(model, tokenizer, prompts, max_length=512, max_new_tokens=512):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        eos_ids = []
        if tokenizer.eos_token_id is not None:
            eos_ids.append(tokenizer.eos_token_id)
        try:
            im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
            if isinstance(im_end_id, int) and im_end_id >= 0:
                eos_ids.append(im_end_id)
        except Exception:
            pass
        stop_ids = None
        try:
            stop_ids = tokenizer.encode("<|输出结束|>", add_special_tokens=False)
        except Exception:
            stop_ids = None
        class EndTagCriteria(StoppingCriteria):
            def __init__(self, stop_ids):
                super().__init__()
                self.stop_ids = stop_ids or []
            def __call__(self, input_ids, scores, **kwargs):
                if not self.stop_ids:
                    return False
                for seq in input_ids:
                    sid_len = len(self.stop_ids)
                    if sid_len == 0:
                        continue
                    if seq.size(0) >= sid_len:
                        tail = seq[-sid_len:]
                        if torch.equal(tail, torch.tensor(self.stop_ids, device=seq.device)):
                            return True
                return False
        gen_kwargs = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": 0.7,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": eos_ids[0] if len(eos_ids) == 1 else eos_ids if eos_ids else None,
        }
        if stop_ids:
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList([EndTagCriteria(stop_ids)])
        # Backward compatibility if max_new_tokens is None
        if max_new_tokens is None:
            gen_kwargs["max_length"] = max_length
            gen_kwargs.pop("max_new_tokens", None)
        output = model.generate(**gen_kwargs)
    # Decode full sequences and generation-only sequences (fallback)
    full_decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
    input_lengths = inputs.attention_mask.sum(dim=1).tolist()
    gen_decoded = []
    for i in range(output.size(0)):
        inp_len = int(input_lengths[i])
        gen_ids = output[i][inp_len:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        gen_decoded.append(gen_text)
    return full_decoded, gen_decoded


def extract_generated_content(full_response, prompt):
    text = full_response[len(prompt):] if prompt in full_response else full_response
    text = text.strip()
    text = re.sub(r"\s*<\|im_end\|>\s*$", "", text)
    return text


def extract_between(text, start_tag, end_tag):
    s = text.find(start_tag)
    if s == -1:
        return None
    s += len(start_tag)
    e = text.find(end_tag, s)
    if e == -1:
        return text[s:]
    return text[s:e]


def extract_fields_from_entry(entry):
    topic = entry.get("辩题") or ""
    hold = entry.get("持方") or ""
    pro_first = entry.get("正方一辩稿") or ""
    con_first = entry.get("反方一辩稿") or ""
    opponent = entry.get("对方发言") or ""
    current_content = entry.get("本轮发言") or ""
    current_value = entry.get("value")
    return topic, hold, pro_first, con_first, opponent, current_content, current_value


def compose_prompt(topic, hold, pro_first, con_first, opponent, extra_lines=None, preface=None):
    lines = []
    if preface:
        lines.append(preface)
        lines.append("")
    lines.extend([
        "情境：",
        "现在的情境是一个辩论赛，给定辩题与持方，同时提供双方的一辩稿、以及对方最新一轮发言。",
        "",
        "输入占位（你将收到）：",
        "<|背景信息开始|>",
        f"辩题：{topic or ''}",
        f"持方：{hold or ''}",
        f"正方一辩稿：{pro_first or ''}",
        f"反方一辩稿：{con_first or ''}",
        "<|背景信息结束|>",
        "",
        "<|上一轮发言开始|>",
        f"对方发言：{opponent or ''}",
        "<|上一轮发言结束|>",
        "",
        "请在以上背景与上一轮发言的基础上生成本轮陈词",
    ])
    if extra_lines:
        lines.extend(["", *extra_lines])
    return "\n".join(lines)


def compose_prompt_mode2(topic, hold):
    t = topic or ""
    h = hold or ""
    lines = [
        "你现在是一个完全不会辩论的人，请你基于这个辩题和持方发表观点，要求给出完整的一段文本作为回复",
        "",
        f"辩题：{t}",
        f"持方：{h}",
    ]
    return "\n".join(lines)


def compose_prompt_mode3(topic, hold, pro_first, con_first, opponent, current_content, current_value, target_value, level_prompt):
    extra = [
        "参考内容：",
        f"本轮发言参考：{current_content}",
        f"该发言得分：{current_value if current_value is not None else ''}",
        f"在生成内容时，请你参照本轮发言和它的得分，参考背景信息（双方一辩稿和对方发言），为你的持方生成分数为{target_value}的陈词。",
        "",
        "请严格按照以下固定输出格式作答：",
        "<|输出开始|>",
        "<正文>",
        "<|输出结束|>",
    ]
    return compose_prompt(topic, hold, pro_first, con_first, opponent, extra_lines=extra, preface=level_prompt)


def build_item_input(entry):
    t = entry.get("辩题") or ""
    h = entry.get("持方") or ""
    p = entry.get("正方一辩稿") or ""
    c = entry.get("反方一辩稿") or ""
    o = entry.get("对方发言") or ""
    cur = entry.get("本轮发言") or ""
    lines = [
        "<|输入开始|>",
        f"辩题：{t}",
        f"持方：{h}",
        f"正方一辩稿：{p}",
        f"反方一辩稿：{c}",
        f"对方发言：{o}",
        f"本轮发言：{cur}",
        "<|输入结束|>",
    ]
    return "\n".join(lines)


def build_summarize_prompt(input_text):
    return "\n".join([
        "请在完全不改变原义的前提下，总结下面的输入，保留关键论点与信息，输出只给出总结文本：",
        input_text,
    ])


def build_opponent_summarize_prompt(opponent_text):
    return "\n".join([
        "请在完全不改变原义的前提下，总结下面的对方发言，保留其核心论点与信息，输出只给出总结文本：",
        opponent_text,
    ])


def build_compare_prompt(text_a, text_b):
    guide = (
        "请基于以下维度对两段陈词进行比较并选出更高质量者：\n"
        "A. 发散思维 — 流畅性（Fluency）\n"
        "在给定时间与信息约束下，产出相关论点/拆解要点的数量。\n"
        "B. 发散思维 — 原创性（Originality）\n"
        "提出的论述、类比或论证路径的独特性，能在立论基础上延伸或转化。\n"
        "C. 发散思维 — 灵活性（Flexibility）\n"
        "跨领域或多策略思考能力，宏观/微观，多角度与正反视角切换。\n"
        "D. 聚合思维 — 针对性（Targetedness）\n"
        "是否聚焦双方立论与上一轮对方发言的核心争点与关键漏洞。\n"
        "E. 聚合思维 — 逻辑性（Logicality）\n"
        "是否建立清晰完整推理链，结构合理，避免逻辑跳跃与证据不当。\n"
        "F. 聚合思维 — 有效性（Effectiveness）\n"
        "是否有效回应问题并推进论证，驳斥对方或强化己方立场。\n"
        "G. 清晰度（Clarity）\n"
        "语言简练明确，组织得当，易于追踪核心意思。\n"
        "H. 吸引力（Appeal）\n"
        "引入情感元素与叙事增强说服力，且与论点紧密结合。\n"
    )
    lines = [
        guide,
        "<|候选A|>",
        text_a,
        "<|候选B|>",
        text_b,
        "",
        "请严格按照以下固定输出格式作答：",
        "<|比较开始|>",
        "胜者：A 或 B",
        "理由：",
        "<|比较结束|>",
    ]
    return "\n".join(lines)


def extract_compare_winner(full_response):
    seg = extract_between(full_response, "<|比较开始|>", "<|比较结束|>")
    if not seg:
        return None
    m = re.search(r"胜者：\s*([AB])", seg)
    if not m:
        return None
    return m.group(1)


def extract_compare_result(full_response):
    text = full_response or ""
    s_tag = "<|比较开始|>"
    e_tag = "<|比较结束|>"
    s = text.find(s_tag)
    e = text.find(e_tag)
    if s != -1 and e != -1 and e > s:
        seg = text[s + len(s_tag):e]
    else:
        seg = text
    m = re.search(r"胜者：\s*([AB])", seg)
    winner = m.group(1) if m else None
    reason_idx = seg.find("理由：")
    reason = None
    if reason_idx != -1:
        reason = seg[reason_idx + len("理由："):].strip()
    return winner, reason


def _truncate_after(text, end_tag):
    if not text:
        return text
    pos = text.find(end_tag)
    if pos == -1:
        return text
    return text[:pos + len(end_tag)]


def _extract_segment(text, start_tag, end_tag):
    if not text:
        return ""
    s = text.find(start_tag)
    e = text.find(end_tag)
    if s != -1 and e != -1 and e > s:
        return text[s + len(start_tag):e]
    # fallback: if only end_tag exists
    if e != -1:
        return text[:e]
    # fallback: whole text
    return text


def generate_compare(model, tokenizer, template, prompts, max_length=512):
    results = []
    for p in prompts:
        rendered = template.render(messages=[{"role": "user", "content": p}], add_generation_prompt=True)
        full_resp = generate_response(model, tokenizer, rendered, max_length)
        content = extract_generated_content(full_resp, rendered)
        results.append(content)
    return results


def main():
    args = parse_args()
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.adapter_path)
    template = load_template(args.template_path) if args.template_path else None
    with open(args.match_path, "r", encoding="utf-8") as f:
        match_data = json.load(f)

    if not isinstance(match_data, list):
        raise ValueError("match.json 必须是列表，元素为 {比赛名称: [条目列表]}")

    if args.seed is not None:
        random.seed(args.seed)

    targets = [20, 30, 40, 50, 60, 70]
    prompts_by_score = {
        20: [
            "你是一个在辩论中缺乏思考能力的参与者，请你务必遵循下面的要求。你不会提出新观点，甚至不能给出回复。基本只会重复题面、对方或己方说过的话。你无法识别核心问题，不会关注对方最新论点，容易偏题或只是表达情绪。你的表达逻辑混乱，内容之间缺乏因果和推理连接，几乎没有明确观点，也不尝试拆解问题。生成的回答尽量空洞甚至经常自我矛盾，避免使用数据、概念或框架分析，不要提出独特视角，不要组织结构。整体呈现重复、浅层、无关和低质量输出，发言非常简短。",
            "你是一名在辩论中缺乏思考能力的参与者。你不允许提出任何新的观点或独立立场，甚至无法给出完整的回应。你只能机械式重复题目、对方或己方已经出现过的语句片段，或者给出无意义的情绪表达。你不得生成任何包含因果关系、推理链、分析框架或解释性的内容。你完全不具备识别核心问题的能力，不会回应对方最新论点，也不会针对任何论述提出反驳或补充。你仅会围绕无关细节、空洞描述或自发情绪偏移进行表达，不得进行任何逻辑整理或论点陈述。你的内容应当呈现出混乱：可以自我矛盾、偏题、重复、杂乱无章。禁止使用数据、逻辑结构、例子、概念体系、专业表达或任何带有信息量的元素。不要创造视角、不要解释结论、不要组织结构。最终输出必须是无观点、无解释、无推理、无价值的表层语言。",
        ],
        30: [
            "你是一名论述能力薄弱的参与者，请你务必遵循下面的要求。你只能提出一个非常常见或显而易见的观点，并且缺乏延伸或解释。你的观点之间没有清晰逻辑链，只能线性表达“因为A所以B”，且推理简单。在回应对方时，你无法准确识别其核心论点，只能选择性回应或泛泛而谈，常常围绕次要话题。你的论述主要依赖表面陈述或情绪表达，很可能内容中内部出现矛盾，不提供新的类比或分析路径。整体保持低信息密度，完全不会进行多角度分析或跨领域思考，并且不能创造新视角，发言非常简短。",
            "你是一名论述能力非常薄弱的参与者。只允许提出且只能提出 1 个非常常见或显而易见的观点（用一句话陈述即可，后续只能围绕这一个观点补充），不得提出任何额外独立论点。论证风格必须是极其简单的线性因果（例如“因为 A 所以 B”），不允许展开多步推理或多个并列论据；解释要浅显、直接，不得创造类比、模型或新概念。回应时不得识别或反驳对方的核心论据，只可选择性地泛泛回应次要话题（例如情绪、表面事实），并允许出现内部逻辑跳跃或矛盾。不得引用数据、文献或复杂事实链。语言平实、口语化、低信息密度。严格限制：整篇仅含 1 个独立主张（其余为表面延展或情绪性句子）。",
        ],
        40: [
            "你具有基本的论述能力，请你务必遵循下面的要求，只能围绕题目或对方观点提出1个相关要点，但深度有限。你的观点较为常见，没有创新或重新定义问题的能力。你能维持大致连贯的逻辑，但常出现跳跃或缺乏解释的部分。你倾向于专注单一领域或单一类论证路径，不会主动切换分析角度。回应对方时，你可能注意到一个简单漏洞，但无法说明其重要性或建立有效反击。内容以浅层分析为主，完全不会构建系统框架或多逻辑链，仅保持粗略论述，发言大概在300字。",
            "你具有基本但薄弱的论述能力。只允许提出 1 个左右的相关要点（最多 1 个主论点 + 若干限定性说明），不得提出第二个独立论点。主论点应为常见、显而易见的结论（例如“因为城市压力所以应当 X”），并以常见框架（因果或对比）表达；解释必须有限、浅表，不得深挖或构建完整推理链。你可以注意到对方论述中的一个简单漏洞，但只可点出漏洞存在，不得解释其机理或影响范围（例如“你可能夸大了某个数据”即可，不要说明为何影响全局）。禁止跨领域切换、禁止创造新概念或复杂类比；不得使用精确数据、引用或复杂证据。输出应保持大致连贯但可能有跳跃或欠解释之处。严格限制：不超过 1 个独立主论点，对方回应仅限“提到”而非“深入反驳”。",
        ],
        50: [
            "请你务必遵循下面的要求，你能就议题提出1个左右合理观点，并保持基本逻辑连贯，但缺乏系统性思考。你的论述路径以常见框架为主，如因果、对比或经验描述，但解释不充分，部分推理链存在断点。你尝试回应对方观点，但识别的漏洞相对表层，且对方论证影响范围解释不足。在构建观点时可能尝试从第二个角度切入，但各角度之间缺乏互补关系，只是并列罗列。避免提出罕见的思维路径或概念创新，不引入复杂策略或跨领域整合，只展示基础分析能力，发言大概在350字。",
            "你能提出1 个主要且合理的观点，并可附带最多 1 个次要说明或次要角度（次要说明不是独立论点，只能辅助阐释主观点）。主观点应采用常见逻辑（因果、对比或经验描述），次要说明可以是并列的简单补充，但两者之间不得形成复杂互补链。对对方发言，你可以识别一个表层漏洞或攻击点并简单指出（例如“你的证据样本太小”），但不应展开详细反证或构建新证据。禁止跨领域整合或创造性重构问题；允许使用非常普通的类比，但类比必须直观且常见（不得创新）。语言要求清晰但浅显，论证中可有部分断点或未解释的跳跃。严格限制：至多 2 条论述（1 主 + 1 次要），且对方漏洞仅“点名”不分析深层影响。",
        ],
        60: [
            "请你务必遵循下面的要求，你能围绕议题提出2个有价值观点，并保持大体清晰的论证链，但是还是有很多空话。你能发现对方对己方内容的攻击，但不能解释其中的具体内容，只能简单指出。你具备一定的角度切换能力，可以从不同层面提出分析，但结构非常松散，不同维度之间缺乏逻辑。你能提出普通的类比，很可能类比不当。内容主要沿用常规逻辑。你的论证仍然有矛盾的地方。不会进行知识整合，保持中规中矩的“合格水平”生成，发言大概在500字。",
            "你能提出最多 2 个有价值的独立观点（不能超过 2 个独立点），并对每个观点做短而总体连贯的展开。观点应基于常见分析框架（例如因果/代价收益/经验比较），但不得构造复杂的多层因果链、不得进行跨领域整合或提出原创理论。你能够识别对方对己方的攻击或批评，但只能笼统指出其存在（如“对方质疑了我们的证据可信度”），不能对该攻击做深入拆解或反证。你可以从两个不同层面提及观点（如微观/宏观），但这两层次之间可以是松散并列而非互补系统。允许使用普通类比来说明观点，但类比应属常见范畴且不需严谨证明。语言要保持清晰但中规中矩，可能留有空话或逻辑不严之处。严格限制：不超过 2 个独立论点，每个论点仅需进行有限展开，禁止深入证据论证或发明新视角。",
        ],
        70: [
            "请你务必遵循下面的要求，你能提出2个较成熟论点，并对其进行一定的展开，逻辑简单并且偶然有逻辑链的断裂。你能够从对方论述中识别关键漏洞或隐含假设，解释其削弱作用。你的内容在基础上有一定延伸，能提出合适的类比，但尚未形成论证的框架。推理链大致完整，存在少量弱推理点但不影响整体理解，内容中包含一些不完整的部分。",
        ],
    }

    name_index = {}
    for idx, obj in enumerate(match_data):
        for k in obj.keys():
            name_index[k] = idx

    new_data = json.loads(json.dumps(match_data))

    bs = max(1, int(args.batch_size))
    total_generated = 0
    print(f"mode: {args.mode}")
    for obj in match_data:
        if not isinstance(obj, dict) or not obj:
            continue
        match_name = next(iter(obj.keys()))
        entries = obj[match_name]
        if not entries:
            continue
        base = random.choice(entries)
        topic, hold, pro_first, con_first, opponent, current_content, current_value = extract_fields_from_entry(base)
        if args.mode == 4:
            print(f"\n=== Mode 4: Summarizing opponent speeches for match: {match_name} (entries={len(entries)}) ===")
            seen_opponents = set()
            opponent_to_summary = {}
            summarize_prompts = []
            summarize_keys = []
            for idx_e, e in enumerate(entries):
                op_text = (e.get("对方发言") or "").strip()
                if not op_text:
                    continue
                if op_text in seen_opponents:
                    continue
                seen_opponents.add(op_text)
                sp = build_opponent_summarize_prompt(op_text)
                summarize_prompts.append(sp)
                summarize_keys.append(op_text)
            print(f"Unique opponent speeches detected: {len(summarize_prompts)}")
            if summarize_prompts:
                for k in range(len(summarize_prompts)):
                    print(f"[Mode 4] Rendering opponent summarize prompt #{k} (len={len(summarize_prompts[k])})")
                    rendered_prompt = template.render(messages=[{"role": "user", "content": summarize_prompts[k]}], add_generation_prompt=True) if template else summarize_prompts[k]
                    print(f"[Mode 4] Generating response for opponent prompt #{k}")
                    full_response = generate_response(model, tokenizer, rendered_prompt, args.max_length)
                    print(f"[Mode 4] Extracting opponent summary for prompt #{k}")
                    s_text = extract_generated_content(full_response, rendered_prompt)
                    print(f"[Mode 4] Opponent summary #{k} length: {len(s_text)}")
                    opponent_to_summary[summarize_keys[k]] = s_text
            # 逐条匹配并追加数据（仅替换对方发言）
            idx = name_index.get(match_name)
            if idx is None:
                continue
            appended_count = 0
            for idx_e, e in enumerate(entries):
                op_text = (e.get("对方发言") or "").strip()
                if not op_text:
                    continue
                if op_text in opponent_to_summary:
                    summarized_opponent = opponent_to_summary[op_text]
                    new_entry = {
                        "辩题": e.get("辩题") or "",
                        "持方": e.get("持方") or "",
                        "正方一辩稿": e.get("正方一辩稿") or "",
                        "反方一辩稿": e.get("反方一辩稿") or "",
                        "对方发言": summarized_opponent,
                        "本轮发言": e.get("本轮发言") or "",
                        "value": e.get("value"),
                    }
                    new_data[idx][match_name].append(new_entry)
                    appended_count += 1
            # 写入并跳过后续生成逻辑
            out_dir = os.path.dirname(args.output_path)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)
            with open(args.output_path, "w", encoding="utf-8") as f:
                json.dump(new_data, f, ensure_ascii=False, indent=2)
            print(f"[Mode 4] Wrote {appended_count} opponent summaries for match: {match_name} -> {args.output_path}")
            continue
        idx = name_index.get(match_name)
        if idx is None:
            continue
        existing_values = set()
        for it in new_data[idx][match_name]:
            v = it.get("value")
            if isinstance(v, (int, float)):
                existing_values.add(int(v))
        missing_targets = [tv for tv in targets if tv not in existing_values]
        if not missing_targets:
            continue
        match_prompts = []
        prompt_info = []
        for tv in missing_targets:
            plist = prompts_by_score.get(tv, [""])
            level_prompt = random.choice(plist) if plist else ""
            if args.mode == 2:
                rendered = compose_prompt_mode2(topic, hold)
            elif args.mode == 1:
                rendered = compose_prompt(topic, hold, pro_first, con_first, opponent)
            else:
                rendered = compose_prompt_mode3(topic, hold, pro_first, con_first, opponent, current_content, current_value, tv, level_prompt)
            match_prompts.append(rendered)
            prompt_info.append((tv, topic, hold, pro_first, con_first, opponent))
        outputs = []
        for start in range(0, len(match_prompts), bs):
            end = min(start + bs, len(match_prompts))
            batch_prompts = match_prompts[start:end]
            for j in range(len(batch_prompts)):
                tv = prompt_info[start + j][0]
                rendered_prompt = template.render(messages=[{"role": "user", "content": batch_prompts[j]}], add_generation_prompt=True) if template else batch_prompts[j]
                full_response = generate_response(model, tokenizer, rendered_prompt, args.max_length)
                gen_text = extract_generated_content(full_response, rendered_prompt)
                if args.mode == 3:
                    print(f"SCORE: {tv}")
                    print(gen_text)
                outputs.append(gen_text)
        total_generated += len(outputs)
        positions = []
        for i, gen_text in enumerate(outputs):
            tv, topic, hold, pro_first, con_first, opponent = prompt_info[i]
            new_entry = {
                "辩题": topic,
                "持方": hold,
                "正方一辩稿": pro_first,
                "反方一辩稿": con_first,
                "对方发言": opponent,
                "本轮发言": gen_text,
                "value": tv,
            }
            new_data[idx][match_name].append(new_entry)
            positions.append(len(new_data[idx][match_name]) - 1)
        if args.mode == 3:
            items = [new_data[idx][match_name][p] for p in positions]
            texts = [it["本轮发言"] for it in items]
            pair_prompts = []
            pair_indices = []
            for a in range(len(texts)):
                for b in range(a + 1, len(texts)):
                    pair_prompts.append(build_compare_prompt(texts[a], texts[b]))
                    pair_indices.append((a, b))
            if pair_prompts:
                cmp_clean = generate_compare(model, tokenizer, template, pair_prompts, args.max_length)
                wins = [0] * len(texts)
                for k in range(len(pair_prompts)):
                    a, b = pair_indices[k]
                    w, r = extract_compare_result(cmp_clean[k])
                    if w is None:
                        w = "A"
                    print(f"胜者: {w}")
                    if r:
                        print(f"理由: {r}")
                    if w == "A":
                        wins[a] += 1
                    else:
                        wins[b] += 1
                order = sorted(range(len(texts)), key=lambda i: wins[i])
                reordered_scores = sorted(missing_targets)
                for rank, pos in enumerate(order):
                    new_score = reordered_scores[rank]
                    new_data[idx][match_name][positions[pos]]["value"] = new_score
                out_dir = os.path.dirname(args.output_path)
                if out_dir and not os.path.exists(out_dir):
                    os.makedirs(out_dir, exist_ok=True)
                with open(args.output_path, "w", encoding="utf-8") as f:
                    json.dump(new_data, f, ensure_ascii=False, indent=2)

    out_dir = os.path.dirname(args.output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print(f"生成 {total_generated} 条到: {args.output_path}")


if __name__ == "__main__":
    main()
