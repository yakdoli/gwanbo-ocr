"""Korean real estate geocoding and stock reference pipeline."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Optional


# ── Korean Address Parser ──────────────────────────────────────────────


@dataclass
class KoreanAddress:
    """Parsed Korean address structure."""

    raw: str = ""
    sido: str = ""  # 서울특별시, 경기도, etc
    sigungu: str = ""  # 강남구, 성남시 분당구, etc
    dong: str = ""  # 역삼동, 서현동, etc
    ri: str = ""  # 리 (rural)
    lot_number: str = ""  # 지번 (번지)
    road_name: str = ""  # 도로명
    building: str = ""  # 건물명/아파트명
    detail: str = ""  # 동/호수

    @property
    def location_id(self) -> str:
        """Generate consistent location ID from parsed address."""
        key = f"{self.sido}|{self.sigungu}|{self.dong}|{self.ri}|{self.lot_number}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    @property
    def display(self) -> str:
        parts = [self.sido, self.sigungu, self.dong, self.ri, self.lot_number]
        return " ".join(p for p in parts if p)


# Address parsing patterns
SIDO_PATTERN = re.compile(
    r"(서울특별시|서울시|서울|부산광역시|부산시|부산|대구광역시|대구시|대구|"
    r"인천광역시|인천시|인천|광주광역시|광주시|광주|대전광역시|대전시|대전|"
    r"울산광역시|울산시|울산|세종특별자치시|세종시|세종|"
    r"경기도|경기|강원도|강원|충청북도|충북|충청남도|충남|"
    r"전라북도|전북|전라남도|전남|경상북도|경북|경상남도|경남|제주특별자치도|제주도|제주)"
)

SIGUNGU_PATTERN = re.compile(
    r"([가-힣]+(?:시|군|구))\s*([가-힣\d]+(?:읍|면|동|가|로|길))?"
)

LOT_PATTERN = re.compile(r"([\d\-]+(?:번지|산)?\s*(?:외\s*\d+필지)?)")


def parse_address(raw: str) -> KoreanAddress:
    """Parse a Korean address string into structured components."""
    addr = KoreanAddress(raw=raw)

    # Extract SIDO
    m = SIDO_PATTERN.search(raw)
    if m:
        addr.sido = m.group(1)

    # Extract SIGUNGU + DONG
    remaining = raw[m.end() :] if m else raw
    m2 = SIGUNGU_PATTERN.search(remaining)
    if m2:
        addr.sigungu = m2.group(1)
        if m2.group(2):
            addr.dong = m2.group(2)

    # Extract lot number
    m3 = LOT_PATTERN.search(raw)
    if m3:
        addr.lot_number = m3.group(1)

    # Extract building name
    bldg_match = re.search(
        r"([가-힣A-Za-z0-9]+(?:아파트|빌라|빌딩|연립|주택|타운))", raw
    )
    if bldg_match:
        addr.building = bldg_match.group(1)

    return addr


# ── Stock Reference ────────────────────────────────────────────────────


@dataclass
class StockInfo:
    """Korean stock reference with corporate events."""

    ticker: str = ""
    company_name_kr: str = ""
    company_name_en: str = ""
    exchange: str = "KRX"
    market: str = ""  # KOSPI / KOSDAQ
    listed_year: int = 0
    delisted_year: int = 0
    events: list[StockEvent] = field(default_factory=list)


@dataclass
class StockEvent:
    """Corporate event affecting stock value/quantity."""

    event_type: str = ""  # split, capital_reduction, capital_increase, ex_dividend, merger, name_change
    event_date: str = ""  # YYYY-MM-DD
    ratio: float = 0.0  # e.g., 5.0 for 5:1 split, 0.5 for 50% reduction
    new_face_value: int = 0
    description: str = ""
    source: str = ""


# Known Korean stock codes and events (sample - would be expanded)
KOREAN_STOCKS: dict[str, StockInfo] = {
    "삼성전자": StockInfo(
        "005930", "삼성전자", "Samsung Electronics", market="KOSPI", listed_year=1975
    ),
    "포항제철": StockInfo(
        "005490",
        "포항제철",
        "POSCO",
        market="KOSPI",
        listed_year=1988,
        events=[StockEvent("name_change", "2002-03-15", 0, 0, "포항제철→POSCO")],
    ),
    "대우중공업": StockInfo(
        "042670", "대우중공업", "", market="KOSPI", listed_year=2000
    ),
    "국민은행": StockInfo(
        "105560", "국민은행", "KB Financial Group", market="KOSPI", listed_year=2001
    ),
    "삼성생명": StockInfo(
        "032830", "삼성생명", "Samsung Life Insurance", market="KOSPI", listed_year=2010
    ),
    "삼성화재": StockInfo(
        "000810", "삼성화재", "Samsung Fire & Marine", market="KOSPI", listed_year=1975
    ),
    "대한항공": StockInfo(
        "003490", "대한항공", "Korean Air", market="KOSPI", listed_year=1966
    ),
    "NAVER": StockInfo(
        "035420", "NAVER", "NAVER Corp", market="KOSPI", listed_year=2008
    ),
    "카카오": StockInfo(
        "035720", "카카오", "Kakao Corp", market="KOSPI", listed_year=2017
    ),
    "SK하이닉스": StockInfo(
        "000660", "SK하이닉스", "SK Hynix", market="KOSPI", listed_year=1996
    ),
    "현대자동차": StockInfo(
        "005380", "현대자동차", "Hyundai Motor", market="KOSPI", listed_year=1974
    ),
    "기아": StockInfo("000270", "기아", "Kia Corp", market="KOSPI", listed_year=1973),
    "LG화학": StockInfo(
        "051910", "LG화학", "LG Chem", market="KOSPI", listed_year=2001
    ),
    "셀트리온": StockInfo(
        "068270", "셀트리온", "Celltrion", market="KOSPI", listed_year=2008
    ),
    "대한비콘": StockInfo("", "대한비콘", "", market="KOSPI"),
    "갑을방직": StockInfo("", "갑을방직", "", market="KOSPI"),
    "한신공업": StockInfo("", "한신공업", "", market="KOSPI"),
    "광덕물산": StockInfo("", "광덕물산", "", market="KOSPI"),
    "대우증권": StockInfo("", "대우증권", "", market="KOSPI"),
    "삼성증권": StockInfo(
        "016360", "삼성증권", "Samsung Securities", market="KOSPI", listed_year=1988
    ),
    "외환은행": StockInfo("004940", "외환은행", "", market="KOSPI"),
    "한국투자증권": StockInfo(
        "071050", "한국투자증권", "Korea Investment Holdings", market="KOSPI"
    ),
    "NH투자증권": StockInfo(
        "005940", "NH투자증권", "NH Investment & Securities", market="KOSPI"
    ),
}


def lookup_stock(name: str) -> Optional[StockInfo]:
    """Look up stock info by Korean company name."""
    for key, info in KOREAN_STOCKS.items():
        if key in name or name in key:
            return info
    return None


def parse_stock_holdings(text: str) -> list[dict]:
    """Parse stock holdings from OCR text like '포항제철 100주, 대우중공업 200주'."""
    holdings = []
    # Pattern: 회사명 + 수량 + 주
    matches = re.findall(
        r"([가-힣A-Za-z&\s]+?)\s*[\(（]?\s*([\d,]+)\s*[\)）]?\s*주", text
    )
    for name, qty in matches:
        name = name.strip()
        if len(name) >= 2:
            info = lookup_stock(name)
            holdings.append(
                {
                    "company_name": name,
                    "ticker": info.ticker if info else "",
                    "shares": int(qty.replace(",", "")),
                    "exchange": info.exchange if info else "KRX",
                }
            )
    return holdings


def geocode_id(address_raw: str) -> tuple[str, KoreanAddress]:
    """Generate unique geocoding ID from address."""
    addr = parse_address(address_raw)
    return addr.location_id, addr
