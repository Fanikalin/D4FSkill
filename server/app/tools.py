import json
import re
from typing import Any, Dict, List, Union

def parse_text(value: str) -> Union[str, List[str]]:
    """
    Разбивает строку по разделителям:
    - '|||' → список элементов
    - '|' → список элементов
    - иначе возвращает строку как есть
    """
    if not value:
        return ""
    if "|||" in value:
        return [part.strip() for part in value.split("|||") if part.strip()]
    if "|" in value:
        return [part.strip() for part in value.split("|") if part.strip()]
    return value.strip()


def parse_block(content: str) -> Any:
    """
    Рекурсивно парсит содержимое тега.
    Возвращает:
    - строку или список строк, если нет вложенных тегов
    - словарь, если есть вложенные теги
    """
    content = content.strip()
    if not content:
        return ""
    # Если нет символа '<' — это простой текст
    if "<" not in content:
        return parse_text(content)

    result = {}
    i = 0
    n = len(content)

    while i < n:
        if content[i] == "<":
            # Находим имя тега (без атрибутов)
            tag_end = content.find(">", i)
            if tag_end == -1:
                break
            tag = content[i+1:tag_end].split()[0]  # отделяем имя от атрибутов
            close_tag = f"</{tag}>"
            # Ищем закрывающий тег с учётом вложенности
            depth = 1
            j = tag_end + 1
            while j < n and depth > 0:
                # Ищем следующее открытие или закрытие
                next_open = content.find(f"<{tag}", j)
                next_close = content.find(close_tag, j)
                if next_close == -1:
                    break
                if next_open != -1 and next_open < next_close:
                    # Встретили вложенный открывающий тег
                    depth += 1
                    j = next_open + len(tag) + 1
                else:
                    depth -= 1
                    if depth == 0:
                        inner = content[tag_end+1:next_close]
                        parsed_inner = parse_block(inner)
                        # Если ключ уже существует, преобразуем в список
                        if tag in result:
                            if not isinstance(result[tag], list):
                                result[tag] = [result[tag]]
                            result[tag].append(parsed_inner)
                        else:
                            result[tag] = parsed_inner
                        j = next_close + len(close_tag)
                        break
                    else:
                        j = next_close + len(close_tag)
            else:
                # Не нашли закрывающий тег, выходим
                break
            i = j
        else:
            # Пропускаем текст вне тегов
            i += 1

    if not result:
        return parse_text(content)
    return result


_TAG_RE = re.compile(r'<(/?)([a-zA-Z_][\w:-]*)\b[^>]*?(/?)>')


def _repair_unclosed_tags(text: str) -> str:
    """
    Модели часто забывают закрыть последний тег вызова инструмента (обычно </tool>).
    Если в тексте есть открытый <tool>, но не хватает закрывающих тегов до конца
    строки, достраивает их в LIFO-порядке, чтобы разбор всё равно нашёл вызов.
    Ничего не меняет, если <tool> в тексте нет вовсе.
    """
    start = text.find('<tool')
    if start == -1:
        return text

    stack = []
    for m in _TAG_RE.finditer(text, start):
        if m.group(3) == '/':  # самозакрывающийся тег <foo/>
            continue
        is_close, tag = m.group(1), m.group(2)
        if not is_close:
            stack.append(tag)
        elif tag in stack:
            while stack and stack[-1] != tag:
                stack.pop()
            if stack:
                stack.pop()

    if not stack:
        return text

    return text + ''.join(f'</{tag}>' for tag in reversed(stack))


def _normalize_call_item(item: Any) -> Union[Dict[str, Any], None]:
    """
    Приводит один элемент JSON-вызова к внутреннему формату {"name": ..., "args": {...}}.
    Понимает как компактную форму {"name": .., "arguments": ..}, так и
    OpenAI-style {"type": "function", "function": {"name": .., "arguments": ..}}.
    """
    if not isinstance(item, dict):
        return None

    func = item.get("function") if isinstance(item.get("function"), dict) else None
    if func is not None:
        name = func.get("name")
        args = func.get("arguments", {})
    elif "name" in item:
        name = item.get("name")
        args = item.get("arguments", item.get("args", {}))
    else:
        return None

    if not name:
        return None
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {}
    if not isinstance(args, dict):
        args = {}
    return {"name": name, "args": args}


def _normalize_tool_json(json_data: Any) -> Any:
    """
    Приводит произвольную JSON-форму вызова инструмента (одиночный объект,
    {"tool_call": {...}}, {"tool_calls": [...]} или просто список) к
    {"name": ..., "args": {...}} либо списку таких словарей. Возвращает None,
    если ничего похожего на вызов инструмента не нашлось.
    """
    if isinstance(json_data, dict) and "tool_call" in json_data:
        json_data = json_data["tool_call"]
    if isinstance(json_data, dict) and isinstance(json_data.get("tool_calls"), list):
        json_data = json_data["tool_calls"]

    if isinstance(json_data, list):
        calls = [c for c in (_normalize_call_item(i) for i in json_data) if c]
        return calls or None
    return _normalize_call_item(json_data)


_INVOKE_OPEN_RE = re.compile(r'<[^<>]*\binvoke\b[^<>]*\bname\s*=\s*"([^"]*)"[^<>]*>', re.IGNORECASE)
_INVOKE_CLOSE_RE = re.compile(r'</[^<>]*\binvoke\b[^<>]*>', re.IGNORECASE)
_INVOKE_BLOCK_RE = re.compile(
    r'<[^<>]*\bcalls?\b[^<>]*>.*?</[^<>]*\bcalls?\b[^<>]*>'
    r'|<[^<>]*\binvoke\b[^<>]*>.*?</[^<>]*\binvoke\b[^<>]*>',
    re.IGNORECASE | re.DOTALL,
)
_PARAM_OPEN_RE = re.compile(r'<[^<>]*\bparameter\b[^<>]*\bname\s*=\s*"([^"]*)"[^<>]*>', re.IGNORECASE)
_PARAM_CLOSE_RE = re.compile(r'</[^<>]*\bparameter\b[^<>]*>', re.IGNORECASE)
# Спецтокены некоторых моделей (в т.ч. родные токены DeepSeek) используют полноширинный
# символ '｜' в качестве разделителя — обычный текст так почти никогда не начинается.
_SPECIAL_TOKEN_TAG_RE = re.compile(r'</?[^<>]*｜[^<>]*>')


def _extract_invoke_style_calls(text: str, lenient: bool = False):
    """
    Некоторые модели вместо канонического <tool> используют похожий на Claude/Continue
    формат <...invoke name="..">...<...parameter name="..">value</...parameter>...</...invoke>,
    иногда обёрнутый в мусорные спецтокены вида <｜｜DSML｜｜ ...｜>. Разбираем его терпимо к
    мусору в имени тега.

    lenient=True: терпим и к недостающему закрывающему </invoke> (для уже завершённого текста —
    восстанавливаем то, что модель забыла закрыть). lenient=False (по умолчанию): требуем реальный
    </invoke>, иначе вызов ещё просто не дописан (стрим в процессе) и не считается найденным —
    иначе живой скрыватель посчитает вызов завершённым раньше времени и хвост потока утечёт
    пользователю как обычный текст.
    Возвращает список {"name": ..., "args": {...}} или None, если ничего похожего нет.
    """
    calls = []
    for inv in _INVOKE_OPEN_RE.finditer(text):
        name = inv.group(1)
        if not name:
            continue
        start = inv.end()
        close = _INVOKE_CLOSE_RE.search(text, start)
        if not close and not lenient:
            continue
        body = text[start:close.start()] if close else text[start:]

        args = {}
        param_opens = list(_PARAM_OPEN_RE.finditer(body))
        for idx, p in enumerate(param_opens):
            pname = p.group(1)
            if not pname:
                continue
            value_start = p.end()
            close_m = _PARAM_CLOSE_RE.search(body, value_start)
            next_open_start = param_opens[idx + 1].start() if idx + 1 < len(param_opens) else len(body)
            value_end = close_m.start() if (close_m and close_m.start() <= next_open_start) else next_open_start
            args[pname] = body[value_start:value_end].strip()

        calls.append({"name": name, "args": args})

    return calls or None


def clean_tools(text: str) -> str:
    """
    Удаляет из текста блоки <tool>...</tool> (и ```tool ... ``` код-блоки) с их
    содержимым, оставляя остальной текст без изменений. Перед этим восстанавливает
    забытые закрывающие теги, чтобы оборванный вызов тоже был вырезан из ответа,
    видимого пользователю.
    Например: 'some text <tool><name>test</name><args>a|||b</args></tool> more' -> 'some text  more'
    """
    text = _repair_unclosed_tags(text)
    text = re.sub(r'```tool\s*\n.*?```', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<tool\b[^>]*>.*?</tool>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = _INVOKE_BLOCK_RE.sub('', text)
    # Что бы ни осталось похожим на спецтокен модели (напр. родные токены DeepSeek с '｜') —
    # это никогда не должно быть показано пользователю, даже если не удалось разобрать вызов.
    return _SPECIAL_TOKEN_TAG_RE.sub('', text)


def _remap_arg_keys_case_insensitive(args: Any, properties: dict) -> Any:
    """
    Если модель написала имя параметра в другом регистре (напр. "filePath" вместо
    "filepath" по схеме инструмента), переименовывает ключ в точное имя из properties.
    Не трогает ключи, которые и так не входят в properties (они просто будут
    проигнорированы ниже, как и раньше).
    """
    if not isinstance(args, dict) or not properties:
        return args
    lower_to_exact = {k.lower(): k for k in properties.keys()}
    remapped = {}
    for key, value in args.items():
        if key in properties:
            remapped[key] = value
        else:
            exact = lower_to_exact.get(key.lower()) if isinstance(key, str) else None
            remapped[exact if exact else key] = value
    return remapped


def _remap_single_unknown_key_to_missing_required(args: Any, properties: dict, required: set) -> Any:
    """
    Модели иногда придумывают собственное имя для параметра вместо того, что указано в схеме
    (напр. "path" вместо "filepath") — это уже не решается сопоставлением без учёта регистра.
    Если после него остаётся РОВНО один ключ аргумента, не входящий в properties, и РОВНО один
    обязательный параметр, которого нет в args — считаем это тем же параметром под другим
    именем и переименовываем. Работает только при полном отсутствии неоднозначности (1:1);
    если совпадений/пропусков больше одного, оставляем как есть — пусть отработает обычная
    валидация с точной диагностикой "не хватает параметра".
    """
    if not isinstance(args, dict):
        return args
    unknown_keys = [k for k in args if k not in properties]
    missing_required = [r for r in required if r not in args]
    if len(unknown_keys) == 1 and len(missing_required) == 1:
        remapped = dict(args)
        remapped[missing_required[0]] = remapped.pop(unknown_keys[0])
        return remapped
    return args


def validate_parameters(args: dict, properties: dict, required: set) -> dict:
    """
    Валидирует и преобразует параметры вызова инструмента согласно схеме.
    Возвращает валидированный словарь args или выбрасывает ValueError.
    """
    # Проверка обязательных
    missing = [req for req in required if req not in args]
    if missing:
        raise ValueError(f"Missing required parameters: {', '.join(missing)}")
    
    validated = {}
    for param_name, param_value in args.items():
        if param_name not in properties:
            # Игнорируем лишние параметры
            continue
        schema = properties[param_name]
        param_type = schema.get("type")
        if param_type == "string":
            if not isinstance(param_value, str):
                validated[param_name] = str(param_value)
            else:
                validated[param_name] = param_value
        elif param_type == "number":
            try:
                validated[param_name] = float(param_value)
            except (ValueError, TypeError):
                raise ValueError(f"Parameter '{param_name}' must be a number")
        elif param_type == "integer":
            try:
                validated[param_name] = int(param_value)
            except (ValueError, TypeError):
                raise ValueError(f"Parameter '{param_name}' must be an integer")
        elif param_type == "boolean":
            if isinstance(param_value, str):
                if param_value.lower() in ("true", "1", "yes"):
                    validated[param_name] = True
                elif param_value.lower() in ("false", "0", "no"):
                    validated[param_name] = False
                else:
                    raise ValueError(f"Parameter '{param_name}' must be a boolean")
            elif isinstance(param_value, bool):
                validated[param_name] = param_value
            else:
                validated[param_name] = bool(param_value)
        elif param_type == "array":
            if isinstance(param_value, str):
                # Разбиваем по разделителям
                if "|||" in param_value:
                    validated[param_name] = [p.strip() for p in param_value.split("|||") if p.strip()]
                elif "|" in param_value:
                    validated[param_name] = [p.strip() for p in param_value.split("|") if p.strip()]
                else:
                    validated[param_name] = [param_value.strip()]
            elif isinstance(param_value, list):
                validated[param_name] = param_value
            else:
                validated[param_name] = [param_value]
        elif param_type == "object":
            # Пока оставляем как есть
            validated[param_name] = param_value
        else:
            # Неизвестный тип - оставляем как есть
            validated[param_name] = param_value
    # Проверяем, что все обязательные присутствуют в validated (уже проверили выше)
    return validated


def tools_to_instructions(tools: Dict[str, Any]) -> str:
    """
    Преобразует словарь инструментов в текстовые инструкции.
    Каждый инструмент представлен как:
    
    Входные данные:
    {
            "type": "function",
            "function": {
                "name": "excel_apply_formula",
                "description": "\nApply Excel formula to cell.\nExcel formula will write to cell with verification.\n",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filepath": {
                            "title": "Filepath",
                            "type": "string"
                        },
                        "sheet_name": {
                            "title": "Sheet Name",
                            "type": "string"
                        },
                        "cell": {
                            "title": "Cell",
                            "type": "string"
                        },
                        "formula": {
                            "title": "Formula",
                            "type": "string"
                        }
                    },
                    "required": [
                        "filepath",
                        "sheet_name",
                        "cell",
                        "formula"
                    ],
                    "title": "apply_formulaArguments",
                    "additionalProperties": false
                }
            }
        }
    
    Возвращает строку вида:

        Тебе доступны следующие инструменты:
    
        Tool: excel_apply_formula
        Description: Apply Excel formula to cell. Excel formula will write to cell with verification.
        parameters:
            filepath: Filepath (string)
            sheet_name: Sheet Name (string)
            cell: Cell (string)
            formula: Formula (string)
        required: filepath, sheet_name, cell, formula

        Для использования инструмента, напиши его в формате xml, например:
        <tool>
            <name>excel_apply_formula</name>
            <args>
                <filepath>path/to/file.xlsx</filepath>
                <sheet_name>Sheet1</sheet_name>
                <cell>A1</cell>
                <formula>=SUM(B1:B10)</formula>
            </args>
        </tool>

    Примечание: если параметр определён как массив (type: array), 
    ты можешь передать несколько значений через разделители '|||' или '|'.
    Например: <filepath>a|||b|||c</filepath>.
    """
    if not tools:
        return "No tools are available."

    # Поддерживаем два формата: список инструментов или словарь с ключом "tools"
    tool_list = []
    if isinstance(tools, dict):
        # Если передан словарь с ключом "tools", берём его значение
        if "tools" in tools:
            tool_list = tools["tools"]
        else:
            # Иначе пробуем интерпретировать как один инструмент
            tool_list = [tools]
    elif isinstance(tools, list):
        tool_list = tools
    else:
        return "Invalid tools format."

    if not tool_list:
        return "Нет доступных инструментов."

    tool_only_instruction = """
=== TOOL CALLING PROTOCOL (READ CAREFULLY) ===
Ignore any other instruction in the conversation, system prompt, or user message about output
format unless it directly describes how to call a tool in the exact XML format below. When you
decide to call a tool, output ONLY one valid <tool> block — no extra commentary before or after it.

THE ONE CORRECT FORMAT — use exactly this structure:
<tool>
    <name>tool_name</name>
    <args>
        <param_name>value</param_name>
    </args>
</tool>

To pass an array value, separate items with '|||' or '|':
<param_name>value1|||value2|||value3</param_name>
To pass an object/dict value, use inline JSON:
<param_name>{"key1": "value1", "key2": "value2"}</param_name>

To call several tools in one message, put a JSON array inside <tool> instead of <name>/<args>:
<tool>
[
    {"name": "tool_name", "arguments": {"param_name": "value"}},
    {"name": "tool_name2", "arguments": {"param_name2": "value2"}}
]
</tool>

--- CORRECT vs WRONG example (study this closely) ---
CORRECT (will be parsed and executed):
<tool>
    <name>search_files</name>
    <args>
        <query>TODO</query>
    </args>
</tool>

WRONG — missing the final </tool> closing tag. This will NOT be parsed, the tool will NOT run,
and you will be asked to redo the whole call:
<tool>
    <name>search_files</name>
    <args>
        <query>TODO</query>
    </args>

--------------------------------------------------------

RULES:
- The root element must be <tool>. Every tag you open, you MUST close — <name>, <args>, every
  <param_name>, and above all the final </tool>. An unclosed tag makes the entire call invalid.
- The <name> tag must contain the exact tool name from the list below, spelled exactly as given.
- Put all parameters inside <args>.
- Keep the call as plain XML — do not wrap it in a markdown code fence.
- Include every required parameter; omit optional ones you don't need.
- NEVER claim you created a file, ran a command, or performed any other action unless you actually
  emitted a matching <tool> call for it (in this turn, or an earlier turn whose result is visible
  in the conversation). Showing a code block with a filename, or describing what a file "will
  contain", is NOT the same as creating it — it creates nothing. If you still need to write
  several files, call the tool for ONE of them now and stop; you will see its result and can call
  the tool for the next one on your following turn. Never summarize multi-step work as "done"
  when you have only actually called the tool for part of it.
- FINAL CHECK before you stop generating: re-read what you just wrote and confirm every tag you
  opened has a matching closing tag, especially </tool> at the very end. A call that looks 95%
  correct but is missing </tool> is treated as no call at all. Also confirm you have not described
  any action as completed that you did not actually call a tool for.
""".strip()

    instructions = ["The following tools are available to you:", tool_only_instruction]

    for tool in tool_list:
        # Извлекаем функцию
        func = tool.get("function", {})
        name = func.get("name", "unknown")
        description = func.get("description", "").strip()
        params = func.get("parameters", {})
        properties = params.get("properties", {})
        required = params.get("required", [])

        # Формируем описание инструмента
        tool_text = []
        tool_text.append(f"\nTool: {name}")
        if description:
            tool_text.append(f"Description: {description}")

        if properties:
            tool_text.append("parameters:")
            for param_name, param_info in properties.items():
                param_type = param_info.get("type", "any")
                param_title = param_info.get("title", param_name)
                param_desc = param_info.get("description", "")
                if param_desc:
                    tool_text.append(f"    {param_name}: {param_title} ({param_type}) - {param_desc}")
                else:
                    tool_text.append(f"    {param_name}: {param_title} ({param_type})")
        else:
            tool_text.append("parameters: none")

        if required:
            tool_text.append(f"required: {', '.join(required)}")
        else:
            tool_text.append("required: none")

        # Пример использования
        example_args = []
        for param_name in required:
            example_args.append(f"                <{param_name}>...</{param_name}>")
        example = "\n".join(example_args)
        tool_text.append(f"""
        Call this tool with:
        <tool>
            <name>{name}</name>
            <args>
{example}
            </args>
        </tool>
        Leave <args></args> empty if the tool takes no parameters. Do not forget </tool> at the end.

        Or call it (possibly together with other tools) using this format in one message:
        <tool>
            [
                {{
                    "name": "{name}",
                    "arguments": {{
{', '.join([f'"{param}": "..."' for param in required])}
                    }}
                }}
            ]
        </tool>
        """)
        instructions.append("\n".join(tool_text))

    return "\n".join(instructions)


def extract_tools(text: str, available_tools: list = None, repair: bool = False) -> Dict[str, Any]:
    """
    Извлекает структурированные данные из XML-подобного текста.

    Формат входа: текст, содержащий корневые теги вида <tag>...</tag>.
    Внутри тегов могут быть вложенные теги и списки через разделители ||| или |.

    Если передан available_tools (список определений инструментов в формате OpenAPI),
    выполняет валидацию: извлекает только те вызовы <tool>, которые соответствуют
    одному из доступных инструментов (по имени и наличию всех обязательных параметров).

    Возвращает словарь с ключом "tool" (список валидных вызовов) или пустой словарь,
    если валидных вызовов нет.

    Помимо канонического <tool>...</tool>, распознаёт также (как страховку от
    отклонений модели от формата, а не как рекламируемую в промпте альтернативу):
    - ```tool ... ``` код-блок с тем же содержимым;
    - "голый" JSON-ответ целиком вида {"name":.., "arguments":..},
      {"tool_call": {...}} или {"tool_calls": [...]} (в т.ч. OpenAI-style
      {"type": "function", "function": {"name":.., "arguments":..}});
    - invoke/parameter-конвенцию (см. _extract_invoke_style_calls).

    repair: если True, недостающие закрывающие теги (</tool>, </invoke>, ...) достраиваются
    "в уме" перед разбором — уместно только когда text это уже ПОЛНОСТЬЮ завершённая генерация
    (конец стрима, нестрим-режим). Держите False (по умолчанию) для проверки ЕЩЁ РАСТУЩЕГО
    буфера во время живого стриминга (см. is_tool_call) — иначе разбор посчитает вызов
    завершённым раньше, чем реально придёт закрывающий тег, и хвост потока (настоящие
    </args>, </tool>, ...) утечёт пользователю как обычный текст вместо того, чтобы быть скрытым.

    Пример:
        text = "<tool><name>test</name><args>a|||b</args></tool>"
        result = extract_tools(text)
        # result == {"tool": {"name": "test", "args": ["a", "b"]}}
    """

    if repair:
        text = _repair_unclosed_tags(text)

    result = {}
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "<":
            tag_end = text.find(">", i)
            if tag_end == -1:
                break
            tag = text[i+1:tag_end]
            close_tag = f"</{tag}>"
            depth = 1
            j = tag_end + 1
            while j < n and depth > 0:
                next_open = text.find(f"<{tag}", j)
                next_close = text.find(close_tag, j)
                if next_close == -1:
                    break
                if next_open != -1 and next_open < next_close:
                    depth += 1
                    j = next_open + len(tag) + 1
                else:
                    depth -= 1
                    if depth == 0:
                        inner = text[tag_end+1:next_close]
                        stripped_inner = inner.strip()
                        if tag == "tool" and (stripped_inner.startswith('[') or stripped_inner.startswith('{')):
                            try:
                                parsed = _normalize_tool_json(json.loads(stripped_inner)) or parse_block(inner)
                            except json.JSONDecodeError:
                                parsed = parse_block(inner)
                        else:
                            parsed = parse_block(inner)
                        # Если ключ уже существует, преобразуем в список
                        if tag in result:
                            if not isinstance(result[tag], list):
                                result[tag] = [result[tag]]
                            result[tag].append(parsed)
                        else:
                            result[tag] = parsed
                        j = next_close + len(close_tag)
                        break
                    else:
                        j = next_close + len(close_tag)
            i = j
        else:
            i += 1

    def _add_tool_calls(parsed):
        if not parsed:
            return
        if "tool" in result:
            existing = result["tool"] if isinstance(result["tool"], list) else [result["tool"]]
            result["tool"] = existing + (parsed if isinstance(parsed, list) else [parsed])
        else:
            result["tool"] = parsed

    # Страховка №1: ```tool ... ``` код-блоки, если канонических <tool> не нашлось
    if "tool" not in result:
        for match in re.finditer(r'```tool\s*\n(.*?)```', text, re.DOTALL | re.IGNORECASE):
            inner = match.group(1).strip()
            try:
                parsed = _normalize_tool_json(json.loads(inner))
            except json.JSONDecodeError:
                parsed = None
            _add_tool_calls(parsed or parse_block(inner))

    # Страховка №1b: модель иногда "галлюцинирует" сырой JSON-ответ API (целое сообщение с
    # tool_calls) и заворачивает его в произвольный код-блок вроде ```json ... ``` вместо
    # реального вызова. Проверяем любой JSON-код-блок — ложных срабатываний почти не бывает,
    # т.к. нужна точная форма {"name":..,"arguments":..} / {"tool_calls":[...]}.
    if "tool" not in result:
        for match in re.finditer(r'```[a-zA-Z0-9_+-]*[ \t]*\n(.*?)```', text, re.DOTALL):
            inner = match.group(1).strip()
            if not (inner.startswith('{') or inner.startswith('[')):
                continue
            try:
                parsed = _normalize_tool_json(json.loads(inner))
            except json.JSONDecodeError:
                parsed = None
            if parsed:
                _add_tool_calls(parsed)

    # Страховка №2: весь ответ — это просто "голый" JSON-вызов без каких-либо тегов
    if "tool" not in result:
        stripped = text.strip()
        if stripped.startswith('{') or stripped.startswith('['):
            try:
                _add_tool_calls(_normalize_tool_json(json.loads(stripped)))
            except json.JSONDecodeError:
                pass

    # Страховка №3: Claude/Continue-подобный формат <invoke name=..><parameter name=..>value
    # (в т.ч. обёрнутый в мусорные спецтокены вида <｜｜DSML｜｜ ...｜>), который некоторые
    # модели используют вместо нашего <tool>, если сорвались на "родной" стиль вызова.
    if "tool" not in result:
        _add_tool_calls(_extract_invoke_style_calls(text, lenient=repair))

    # Если нет доступных инструментов, возвращаем как есть
    if available_tools is None:
        return result

    # Построим карту доступных инструментов: имя -> {required: set, properties: {...}}
    tools_map = {}
    for tool_def in available_tools:
        func = tool_def.get("function", {})
        name = func.get("name")
        if name:
            params = func.get("parameters", {})
            required = set(params.get("required", []))
            properties = params.get("properties", {})
            tools_map[name] = {"required": required, "properties": properties}

    # Фильтруем вызовы <tool>
    valid_calls = []
    if "tool" in result:
        tool_data = result["tool"]
        if not isinstance(tool_data, list):
            tool_data = [tool_data]
        for call in tool_data:
            # call должен быть словарём с ключами name и args
            if not isinstance(call, dict):
                continue
            call_name = call.get("name")
            if call_name not in tools_map:
                continue
            tool_info = tools_map[call_name]
            required = tool_info["required"]
            properties = tool_info["properties"]
            args = call.get("args", {})
            # args может быть строкой (если не распарсился как словарь)
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except:
                    args = {}
            # Модели иногда путают регистр имени параметра (напр. "filePath" вместо
            # "filepath" по схеме) — сопоставляем без учёта регистра прежде, чем отбрасывать
            # вызов как "не хватает обязательного параметра".
            args = _remap_arg_keys_case_insensitive(args, properties)
            args = _remap_single_unknown_key_to_missing_required(args, properties, required)
            # Валидация параметров по схеме
            try:
                validated_args = validate_parameters(args, properties, required)
                valid_calls.append({
                    "name": call_name,
                    "args": validated_args
                })
            except ValueError:
                continue

    # Возвращаем только валидные вызовы (всегда список)
    if valid_calls:
        return {"tool": valid_calls}
    else:
        return {}


def is_tool_call(text: str, available_tools: list) -> bool:
    """
    Проверяет, является ли текст корректным вызовом инструмента.
    Возвращает True, если текст содержит хотя бы один валидный <tool> вызов,
    соответствующий одному из доступных инструментов.
    """
    extracted = extract_tools(text, available_tools)
    return "tool" in extracted and bool(extracted["tool"])


def is_special_token_start(text: str) -> bool:
    """
    Дёшево детектирует потенциальное начало вставки спецтокенов модели (обёрнутых в
    полноширинный '｜', как у нативных токенов DeepSeek). Нужна посимвольному скрывателю
    при стриминге, чтобы прятать такой мусор в реальном времени, даже когда внутри не наш
    канонический <tool>-формат, а что-то вроде <｜｜DSML｜｜ invoke ...>.
    Ограничена по длине, чтобы ложное срабатывание не проглотило весь легитимный ответ.
    """
    if not text.startswith('<'):
        return False
    if '｜' in text[:3]:
        return len(text) < 4000
    return len(text) <= 2


def is_tool_call_start(text: str) -> bool:
    """
    Проверяет, является ли строка началом вызова инструмента.
    Возвращает True, если строка может быть префиксом корректного вызова
    в формате <tool>...<name>...</name>... (с возможными пробелами).
    Иначе False.
    Примеры:
      '' -> True
      '<' -> True
      '<t' -> True
      '<to' -> True
      '<too' -> True
      '<tool' -> True
      '<tool>' -> True
      '<tool> ' -> True
      '<tool> <' -> True
      '<tool> <n' -> True
      '<tool> <na' -> True
      '<tool> <nam' -> True
      '<tool> <name' -> True
      '<tool> <name>' -> True
      '<tool> <name>func' -> True
      '<tool> <name>func</' -> True
      '<tool> <name>func</n' -> True
      '<tool> <name>func</na' -> True
      '<tool> <name>func</nam' -> True
      '<tool> <name>func</name' -> True
      '<tool> <name>func</name>' -> True
      '<tool> <name>func</name> ' -> True
      '<tool> <name>func</name> \n<value>1</value></tool>' -> True
      '<tool> <name>func</name> hello' -> False
      '<tool>hello' -> False
      '<tool><name>execute</name> hello' -> False
    """
    # Пустая строка — префикс любого вызова
    if text == "":
        return True

    if not text.startswith("<"):
        return False

    # Ищем первый '>'
    gt_pos = text.find('>')
    if gt_pos == -1:
        # Нет закрывающего '>', проверяем, что текст является префиксом "<tool"
        # т.е. каждый символ после '<' должен совпадать с началом "tool"
        tool = "tool"
        for i, ch in enumerate(text[1:]):
            if i >= len(tool) or ch != tool[i]:
                return False
        return True

    # Есть '>', проверяем, что до него ровно "<tool" (без пробелов внутри)
    if text[:gt_pos] != "<tool":
        return False

    # Остаток после '>'
    rest = text[gt_pos+1:]
    stripped_rest = rest.lstrip()

    if not stripped_rest:
        return True

    # Если начинается с '[', это JSON-массив — допустимое начало
    if stripped_rest.startswith('['):
        return True

    if not stripped_rest.startswith("<"):
        # После '>' идёт текст, не являющийся XML-тегом и не JSON — не начало
        return False

    # Теперь нужно проверить, что stripped_rest является префиксом "<name>...",
    # то есть после '<' идёт последовательность символов, которая является
    # префиксом "name" (или уже содержит закрывающий '>' и имя равно "name").
    # Найдём позицию '>' внутри stripped_rest (если есть)
    inner_gt = stripped_rest.find('>')
    if inner_gt == -1:
        # Нет закрывающего '>' внутри, значит это префикс "<name"
        # Проверяем, что все символы после '<' являются префиксом "name"
        name = "name"
        for i, ch in enumerate(stripped_rest[1:]):
            if i >= len(name) or ch != name[i]:
                return False
        return True
    else:
        # Есть закрывающий '>', имя должно быть ровно "name"
        if stripped_rest[1:inner_gt] != "name":
            return False
        # Теперь проверяем закрывающий </name>
        close_tag = "</name>"
        close_pos = stripped_rest.find(close_tag, inner_gt)
        if close_pos == -1:
            # Нет закрывающего — это начало, допустимо
            return True
        # После закрывающего разрешены только пробелы или '<' (продолжение)
        after = stripped_rest[close_pos + len(close_tag):]
        stripped_after = after.lstrip()
        if not stripped_after:
            return True
        if stripped_after.startswith("<"):
            return True
        return False


# Многие агентные фреймворки (в т.ч. Continue) сами учат модель писать код-блоки вида
# "```python main.py" с плейсхолдером "// ... existing code ..." для ДЕМОНСТРАЦИИ, явно
# оговаривая "for implementing changes, use the edit tools" — но модель на практике часто
# путает это с реальным применением изменения. Это самая частая причина, по которой модель
# заявляет "готово", не вызвав ни одного инструмента.
_CODE_FENCE_WITH_FILENAME_RE = re.compile(
    r'```[ \t]*(?:[a-zA-Z0-9_+-]+[ \t]+)?[\w./\\-]+\.[a-zA-Z0-9]{1,10}[ \t]*\n'
)
_EXISTING_CODE_PLACEHOLDER_RE = re.compile(
    r'(#|//|<!--)\s*\.\.\.\s*existing code\s*\.\.\.', re.IGNORECASE
)


def looks_like_undone_edit(text: str) -> bool:
    """
    True, если текст похож на "описание" изменения (код-блок с именем файла в info-строке,
    или плейсхолдер вида "// ... existing code ...") вместо реального вызова инструмента.
    Такой код-блок никуда не применяется.
    """
    return bool(_CODE_FENCE_WITH_FILENAME_RE.search(text)) or bool(_EXISTING_CODE_PLACEHOLDER_RE.search(text))


def diagnose_tool_call_issue(text: str, available_tools: list) -> str:
    """
    Вызывается только когда extract_tools() не нашёл ни одного валидного вызова.
    Возвращает короткую машинную диагностику причины, чтобы ретрай мог дать модели
    точечную подсказку вместо общего "ты не использовала инструмент".
    """
    if looks_like_undone_edit(text):
        return ("described_not_executed: you described a file change as a fenced code block "
                "(with a filename in the info string, e.g. ```python main.py) or with an "
                "\"... existing code ...\" placeholder, instead of actually calling a tool. That "
                "code block is NOT applied anywhere and the user cannot see it as a diff — nothing "
                "happens until you emit a real <tool> call (e.g. edit_file or create_new_file) with "
                "the tool name and the COMPLETE file contents as arguments. Call the appropriate "
                "tool now instead of showing the change as text.")

    has_attempt = any(marker in text for marker in ('<tool', '```tool', '"tool_call', '"function"')) \
        or bool(_INVOKE_OPEN_RE.search(text)) or '｜' in text
    if not text or not has_attempt:
        return "no_attempt: you did not attempt to call a tool at all — no <tool> block was found in your output."

    # Разбираем без фильтрации по available_tools, чтобы увидеть, что вообще распозналось
    raw = extract_tools(text, available_tools=None, repair=True)
    tool_data = raw.get("tool")

    if not tool_data:
        if '<tool' in text and '</tool>' not in text:
            return ("unclosed_tag: you opened a <tool> block but never wrote the matching closing </tool> tag, "
                    "and the content inside was too broken to recover automatically. Rewrite the SAME call from "
                    "scratch and make sure every tag you open (<tool>, <args>, ...) is closed, especially </tool> "
                    "at the very end.")
        if '｜' in text or _INVOKE_OPEN_RE.search(text):
            return ("wrong_format: you used a different tool-calling convention (special tokens / "
                    "invoke-parameter style) instead of the one described above. Only the <tool><name>...</name>"
                    "<args>...</args></tool> XML format is understood here. Rewrite the call using exactly that "
                    "format and nothing else.")
        return ("invalid_json: the content of your tool call could not be parsed as valid XML or JSON. Rewrite it "
                "from scratch, double-checking brackets, quotes and tag names.")

    calls = tool_data if isinstance(tool_data, list) else [tool_data]
    tools_map = {}
    for tool_def in (available_tools or []):
        func = tool_def.get("function", {})
        if func.get("name"):
            tools_map[func["name"]] = func

    reasons = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        name = call.get("name")
        if name not in tools_map:
            reasons.append(f"unknown_tool: '{name}' does not match the name of any available tool exactly.")
            continue
        func = tools_map[name]
        params = func.get("parameters", {})
        required = set(params.get("required", []))
        properties = params.get("properties", {})
        args = call.get("args", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        try:
            validate_parameters(args if isinstance(args, dict) else {}, properties, required)
        except ValueError as e:
            reasons.append(f"missing_required: {e}")

    if reasons:
        return " | ".join(dict.fromkeys(reasons))
    return ("unclosed_tag: the tool call was close to valid but still got rejected — re-check that every tag is "
            "closed and that the tool name matches one from the list exactly.")
