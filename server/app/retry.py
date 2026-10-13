import re

# DeepSeek отдаёт ошибки (лимиты, "сервер занят", ...) прямо в тексте ответа, одним обычным
# чанком — для клиента они неотличимы от валидного ответа. Определяем их по тексту.
ERROR_PATTERNS = [re.compile(p, re.IGNORECASE) for p in (
    r"please try again (later|in a)",
    r"try again later",
    r"message is being generated",
    r"too many requests",
    r"too many (concurrent|simultaneous)",
    r"rate.?limit",
    r"server is (busy|overloaded|currently)",
    r"servers? (are|is) busy",
    r"service (is )?(temporarily )?(unavailable|busy)",
    r"temporarily unavailable",
    r"request (was )?(blocked|rejected|failed)",
    r"something went wrong",
    r"an error occurred",
    r"internal server error",
    r"bad gateway|gateway time-?out",
    r"you('ve| have) (sent|made) too many",
    r"exceeded .{0,40}(limit|quota)",
    r"服务器繁忙",
    r"请稍后(再试|重试)",
    r"请求过于频繁",
    r"消息正在生成",
    r"сервер(ы)? (занят|перегружен)",
    r"слишком много запросов",
    r"попробуйте (еще|ещё) раз (позже|позднее)",
)]

# Сообщение об ошибке всегда короткое. Длинный ответ, где просто упомянут, скажем, "rate limit"
# (например, объяснение про rate limiting), ошибкой не считаем.
ERROR_MAX_LENGTH = 300


def is_error_text(text: str) -> bool:
    """True, если text — пустой/состоит из пробелов либо похож на сообщение об ошибке DeepSeek."""
    stripped = (text or "").strip()
    if not stripped:
        return True
    if len(stripped) > ERROR_MAX_LENGTH:
        return False
    return any(p.search(stripped) for p in ERROR_PATTERNS)
