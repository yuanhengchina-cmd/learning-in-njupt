import re
import os

# ============================================================
# 单词-类型关系表（根据你的源程序设计，可自行扩展）
# ============================================================
KEYWORDS = {
    'void':   1,
    'main':   2,
    'int':    3,
    'cout':   4,
    'return': 5,
}

OPERATORS = {
    '(':  6,
    ')':  7,
    '{':  8,
    '}':  9,
    '<<': 10,
    ';':  11,
    ',':  12,
    '=':  13,
    '*':  14,
    '/':  15,
}

TYPE_IDENTIFIER = 16   # 合法标识符
TYPE_INTEGER    = 17   # 十进制整数
TYPE_OCT        = 18   # 八进制整数（提高要求）
TYPE_HEX        = 19   # 十六进制整数（提高要求）

# ============================================================
# 第一步：读入源程序，并记录行号列号（用于提高要求）
# ============================================================
def read_source(filepath: str) -> str:
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


# ============================================================
# 第二步：预处理
#   - 删除 // 行注释
#   - 删除 /* */ 块注释（可跨行）
#   - 将换行/制表符替换为单个空格
#   - 合并多余的连续空格（但保留单个空格）
# ============================================================
def preprocess(source: str) -> str:
    # 删除块注释 /* ... */（非贪婪，支持跨行）
    source = re.sub(r'/\*.*?\*/', ' ', source, flags=re.DOTALL)
    # 删除行注释 // ...
    source = re.sub(r'//[^\n]*', ' ', source)
    # 将换行、制表符统一替换为空格
    source = re.sub(r'[\r\n\t]+', ' ', source)
    # 合并连续空格为一个
    source = re.sub(r' +', ' ', source)
    # 去掉首尾空格
    source = source.strip()
    return source


# ============================================================
# 第三步：词法分析
#   - 识别关键字、标识符、整数（十进制/八进制/十六进制）、运算符
#   - 对非法标识符（数字开头后跟字母/数字）跳过但记录
# ============================================================
def tokenize(preprocessed: str, original_source: str):
    """
    返回：
        tokens       - 合法单词列表，每项为 (序号, 类别码, 值)
        illegal_list - 非法标识符列表，每项为 (序号, 值, 行号, 列号)
    """
    tokens = []
    illegal_list = []
    word_index = 0       # 单词序号（包含非法词）
    i = 0
    n = len(preprocessed)

    # 建立原始源程序的位置映射（提高要求：行列号）
    # 由于预处理后已失去行列信息，需在原始源程序中查找
    def find_position(token_val: str) -> tuple:
        """在原始源程序中找到 token_val 第一次出现的位置（行, 列）"""
        for line_no, line in enumerate(original_source.splitlines(), 1):
            col = line.find(token_val)
            if col != -1:
                return line_no, col + 1
        return -1, -1

    while i < n:
        # 跳过空格
        if preprocessed[i] == ' ':
            i += 1
            continue

        # ---- 尝试匹配双字符运算符（如 <<）----
        if i + 1 < n and preprocessed[i:i+2] in OPERATORS:
            op = preprocessed[i:i+2]
            word_index += 1
            tokens.append((word_index, OPERATORS[op], op))
            i += 2
            continue

        # ---- 尝试匹配单字符运算符 ----
        if preprocessed[i] in OPERATORS:
            op = preprocessed[i]
            word_index += 1
            tokens.append((word_index, OPERATORS[op], op))
            i += 1
            continue

        # ---- 数字开头：可能是整数，也可能是非法标识符 ----
        if preprocessed[i].isdigit():
            j = i
            # 收集连续的字母和数字
            while j < n and (preprocessed[j].isalnum() or preprocessed[j] == '_'):
                j += 1
            token_val = preprocessed[i:j]
            word_index += 1

            # 判断是否是纯数字（合法整数）
            if re.fullmatch(r'0[xX][0-9a-fA-F]+', token_val):
                # 十六进制整数（提高要求）
                tokens.append((word_index, TYPE_HEX, token_val))
            elif re.fullmatch(r'0[0-7]+', token_val):
                # 八进制整数（提高要求）
                tokens.append((word_index, TYPE_OCT, token_val))
            elif re.fullmatch(r'[0-9]+', token_val):
                # 十进制整数
                tokens.append((word_index, TYPE_INTEGER, token_val))
            else:
                # 非法标识符（数字开头后跟字母）
                line_no, col_no = find_position(token_val)
                illegal_list.append((word_index, token_val, line_no, col_no))
                # 跳过，不加入 tokens

            i = j
            continue

        # ---- 字母或下划线开头：标识符或关键字 ----
        if preprocessed[i].isalpha() or preprocessed[i] == '_':
            j = i
            while j < n and (preprocessed[j].isalnum() or preprocessed[j] == '_'):
                j += 1
            token_val = preprocessed[i:j]
            word_index += 1

            if token_val in KEYWORDS:
                tokens.append((word_index, KEYWORDS[token_val], token_val))
            else:
                tokens.append((word_index, TYPE_IDENTIFIER, token_val))

            i = j
            continue

        # ---- 其他字符：跳过（可根据需要扩展处理）----
        i += 1

    return tokens, illegal_list


# ============================================================
# 输出与保存
# ============================================================
def display_and_save(preprocessed: str, tokens: list, illegal_list: list,
                     source_dir: str):
    """
    1. 打印预处理结果
    2. 打印词法分析结果
    3. 将两者分别保存到 txt 文件
    """

    # --- 预处理结果 ---
    print("=" * 60)
    print("【预处理后的字符流】")
    print("=" * 60)
    print(preprocessed)
    print()

    preprocessed_path = os.path.join(source_dir, 'preprocessed.txt')
    with open(preprocessed_path, 'w', encoding='utf-8') as f:
        f.write(preprocessed)
    print(f"预处理结果已保存到：{preprocessed_path}\n")

    # --- 词法分析结果 ---
    print("=" * 60)
    print("【词法分析结果】")
    print("=" * 60)
    header = f"{'单词序号':<8}{'单词类型':<10}{'单词的值'}"
    print(header)
    print("-" * 40)

    lines_out = [header, "-" * 40]
    for seq, ttype, val in tokens:
        row = f"{seq:<8}{ttype:<10}{val}"
        print(row)
        lines_out.append(row)

    # 非法标识符信息
    print()
    lines_out.append("")
    if illegal_list:
        for seq, val, line_no, col_no in illegal_list:
            msg = f"发现非法标识符：{val}，位于源程序第 {seq} 个单词"
            pos_msg = f"  （原始文件位置：第 {line_no} 行，第 {col_no} 列）"
            print(msg)
            print(pos_msg)
            lines_out.append(msg)
            lines_out.append(pos_msg)
    else:
        msg = "未发现非法标识符"
        print(msg)
        lines_out.append(msg)

    tokens_path = os.path.join(source_dir, 'tokens.txt')
    with open(tokens_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines_out))
    print(f"\n词法分析结果已保存到：{tokens_path}")


# ============================================================
# 主函数
# ============================================================
def main():
    source_file = 'source.txt'   # 修改为你的源程序路径

    if not os.path.exists(source_file):
        print(f"错误：找不到文件 {source_file}")
        return

    source_dir = os.path.dirname(os.path.abspath(source_file))
    original_source = read_source(source_file)

    # 第二步：预处理
    preprocessed = preprocess(original_source)

    # 第三步：词法分析
    tokens, illegal_list = tokenize(preprocessed, original_source)

    # 输出并保存结果
    display_and_save(preprocessed, tokens, illegal_list, source_dir)


if __name__ == '__main__':
    main()