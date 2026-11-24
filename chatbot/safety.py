"""Deterministic safety checks for FinSight's financial-advice boundary.

This module is independent of LangChain and Groq so its policy can be tested
without an API key or model dependency.
"""

import re
import unicodedata


FINANCIAL_ADVICE_REFUSAL = (
    "I'm sorry, but Clarry does not provide financial or investment advice. "
    "I can help you view and understand the information recorded in FinSight, "
    "but I cannot recommend what you should buy, sell, hold, invest in, or "
    "otherwise do with your money. Any financial decision is your own "
    "responsibility. Please consult a suitably qualified professional for "
    "personal advice. FinSight is a tracking and analysis tool, not a "
    "financial adviser."
)

_FINANCIAL_TOPIC = re.compile(
    r"\b(?:money|finance|financial|budget|spend(?:ing)?|expenses?|"
    r"sav(?:e|ing|ings)|invest(?:ment|ments|ing)?|stocks?|shares?|"
    r"share market|mutual funds?|etfs?|exchange traded funds?|bonds?|"
    r"crypto(?:currency)?|bitcoin|portfolio|sip|ipo|nfo|fixed deposits?|"
    r"fds?|gold|insurance|loans?|credit|debt|retirement|pension|"
    r"asset allocation|property|real estate|house|home|car|credit score)\b",
    re.IGNORECASE,
)

_DIRECT_ADVICE_REQUEST = re.compile(
    r"\b(?:should i|what should i|what would you (?:do|recommend)|"
    r"what do you recommend|(?:can|could|would) you recommend|"
    r"recommend (?:me|a|an|the)|suggest (?:me|a|an|the)|"
    r"(?:can|could|would) you suggest|what would you suggest|"
    r"advise me|how should i|"
    r"give me (?:some )?(?:financial |investment )?advice|"
    r"provide (?:some )?(?:financial |investment )?advice|"
    r"which .{0,60} should i|where should i|"
    r"how much should i (?:invest|save|spend|borrow|allocate)|"
    r"is it (?:worth|wise|safe) (?:for me )?(?:to )?(?:buy|sell|invest|switch)|"
    r"is .{1,80} a good (?:investment|buy)|"
    r"tell me (?:what|where|which) .{0,80} (?:to buy|to sell|to invest|i should)|"
    r"what is the best way to (?:save|invest|grow my money|manage my money)|"
    r"is it a good idea to (?:buy|sell|invest)|"
    r"(?:financial|investment) recommendations?|stock picks?|fund picks?"
    r")\b",
    re.IGNORECASE,
)

_SAVING_OR_MONEY_TIPS_REQUEST = re.compile(
    r"\b(?:tips? (?:for|to) (?:saving|save|investing|invest|reducing|cutting)|"
    r"ways to (?:save|invest|grow my money|reduce (?:my )?spending|"
    r"pay off (?:my )?debt)|"
    r"how (?:can|do) i (?:save|invest|grow my money|manage my money|"
    r"reduce (?:my )?(?:expenses|spending)|pay off (?:my )?debt)|"
    r"how to (?:start )?(?:saving|save|invest|investing|grow my money|"
    r"build wealth|reduce (?:my )?(?:expenses|spending))|"
    r"tell me how to (?:save|invest|grow my money)|"
    r"tell me where to invest|"
    r"(?:saving|investment|financial) tips|"
    r"help me (?:save|invest|reduce (?:my )?(?:expenses|spending)|manage my money)|"
    r"how (?:can|do) i (?:improve|raise) my credit score|"
    r"how to (?:get out of|manage) (?:my )?debt"
    r")\b",
    re.IGNORECASE,
)

_BEST_PRODUCT_REQUEST = re.compile(
    r"\b(?:best|top|good|high-return|highest-return)\s+(?:.{0,30}\s)?(?:stocks?|shares?|mutual funds?|"
    r"etfs?|bonds?|cryptocurrencies|crypto|sip|ipo|nfo|investments?|"
    r"fixed deposits?|fds?|insurance|loans?|credit cards?|"
    r"investment options?|funds? to invest|funds?)\b",
    re.IGNORECASE,
)

_PRODUCT_EVALUATION_REQUEST = re.compile(
    r"\b(?:is|are)\b.{1,100}\b(?:worth buying|worth investing|a good buy|"
    r"a good investment|good for me|right for me|suitable for me|"
    r"best for me|a good choice|safe to invest|a safe investment|"
    r"a good time to (?:buy|sell|invest))\b",
    re.IGNORECASE,
)

_BEST_OR_COMPARISON_CHOICE = re.compile(
    r"\b(?:which|what)\b.{0,80}\b(?:better|best|good|right for me|"
    r"should i choose|should i pick)\b",
    re.IGNORECASE,
)

_INVESTMENT_PRODUCT = re.compile(
    r"\b(?:stocks?|shares?|mutual funds?|etfs?|bonds?|crypto(?:currency)?|"
    r"bitcoin|sip|ipo|nfo|fixed deposits?|gold|insurance|loans?|"
    r"credit cards?|portfolio|investments?|investment funds?|funds?)\b",
    re.IGNORECASE,
)

_MARKET_PREDICTION_REQUEST = re.compile(
    r"\b(?:predict|forecast|price target|will .{1,60}"
    r"(?:stock|share|market|bitcoin|crypto|nifty|sensex)|"
    r"where will .{1,60} (?:trade|be))\b",
    re.IGNORECASE,
)

_PORTFOLIO_ADVICE_REQUEST = re.compile(
    r"\b(?:review|analyse|analyze|evaluate|rate|optimise|optimize|rebalance|"
    r"build|create|design)\b.{0,100}\bportfolio\b",
    re.IGNORECASE,
)


_ACTION_AND_PRODUCT = re.compile(
    r"\b(?:buy|sell|hold|invest in|put (?:your|my) money in|"
    r"switch to|move (?:your|my) money to|allocate to|choose|pick)\b"
    r".{0,100}\b(?:stocks?|shares?|mutual funds?|etfs?|bonds?|"
    r"crypto(?:currency)?|bitcoin|sip|ipo|nfo|fixed deposits?|gold|"
    r"insurance plans?|loans?|credit cards?)\b",
    re.IGNORECASE | re.DOTALL,
)

_OUTPUT_RECOMMENDATION = re.compile(
    r"\b(?:i recommend|i suggest|my recommendation is|you should|"
    r"you ought to|you may want to|you could|consider|try|the best option is|"
    r"the best choice is|you can)\b.{0,180}\b(?:buy|sell|hold|invest(?:ing)?|"
    r"save|spend|borrow|repay|pay off|reduce|increase|allocate|switch|"
    r"choose|pick|stocks?|shares?|mutual funds?|etfs?|bonds?|crypto|"
    r"sip|ipo|nfo|money|budget|savings?|debt|loan|insurance|portfolio)\b",
    re.IGNORECASE | re.DOTALL,
)


def _normalise(text: str) -> str:
    """Normalise Unicode and whitespace before applying policy patterns."""
    value = unicodedata.normalize("NFKC", str(text or ""))
    return re.sub(r"\s+", " ", value).strip()


def is_financial_advice_request(message: str, context: str = "") -> bool:
    """Identify requests for personalised money decisions or recommendations.

    Context is used only to identify the subject of short follow-ups such as
    "what should I do with it?". Neutral definitions and factual app queries
    are intentionally allowed.
    """
    text = _normalise(message)
    if not text:
        return False

    context_text = _normalise(context)
    has_financial_topic = bool(
        _FINANCIAL_TOPIC.search(text) or _FINANCIAL_TOPIC.search(context_text)
    )

    if _BEST_PRODUCT_REQUEST.search(text):
        return True
    if _PRODUCT_EVALUATION_REQUEST.search(text) and has_financial_topic:
        return True
    if (
        _BEST_OR_COMPARISON_CHOICE.search(text)
        and (
            _INVESTMENT_PRODUCT.search(text)
            or _INVESTMENT_PRODUCT.search(context_text)
        )
    ):
        return True
    if _MARKET_PREDICTION_REQUEST.search(text) and has_financial_topic:
        return True
    if _PORTFOLIO_ADVICE_REQUEST.search(text):
        return True
    if _DIRECT_ADVICE_REQUEST.search(text) and has_financial_topic:
        return True
    if _SAVING_OR_MONEY_TIPS_REQUEST.search(text) and has_financial_topic:
        return True
    if _ACTION_AND_PRODUCT.search(text):
        return True

    # Handle short or open-ended advice requests using recent conversation context.
    if (
        re.search(
            r"\b(?:what should i do|what do i do|what would you do|"
            r"which one should i choose|which one should i pick)\b",
            text,
            re.IGNORECASE,
        )
        and has_financial_topic
    ):
        return True

    # Affordability questions ask the chatbot to make a personal decision.
    if re.search(
        r"\b(?:can i afford|should i spend|can i spend|"
        r"can i (?:take|apply for) (?:out )?(?:a )?loan)\b",
        text,
        re.I,
    ):
        return True

    return False


def contains_financial_recommendation(response: str) -> bool:
    """Detect advice-like financial recommendations in a generated answer."""
    text = _normalise(response)
    if not text:
        return False

    # The fixed refusal intentionally names prohibited actions and is safe.
    if text.startswith(
        "I'm sorry, but Clarry does not provide financial or investment advice."
    ):
        return False

    # A referral to a qualified professional is allowed, not a recommendation
    # about which financial action or product the user should choose.
    text = re.sub(
        r"\b(?:consult|speak with|talk to|contact|seek)\b.{0,80}"
        r"\b(?:qualified|licensed|registered|professional|adviser|advisor)\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    return bool(
        _OUTPUT_RECOMMENDATION.search(text)
        or _ACTION_AND_PRODUCT.search(text)
    )
