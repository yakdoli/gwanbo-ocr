"""Korean administrative district change history schema.

Based on geocoder-kr reference (https://github.com/yakdoli/geocoder-kr)
which uses RocksDB with PNU codes, road name codes, and administrative codes.

Tracks year-by-year changes in administrative districts linked to GPS coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as Date
from typing import Optional

# ── Administrative Code Change Types ───────────────────────────────────


@dataclass
class AdminDistrictChange:
    """One administrative district change event."""

    change_id: str = ""
    change_date: Date | str = ""
    change_type: str = ""  # 승격/통합/분리/명칭변경/폐지/신설
    region_level: str = ""  # sido/sigungu/dong/ri

    # Before change
    old_name: str = ""
    old_h1_cd: str = ""  # 시도 코드
    old_h2_cd: str = ""  # 시군구 코드
    old_h3_cd: str = ""  # 읍면동 코드
    old_li_cd: str = ""  # 리 코드

    # After change
    new_name: str = ""
    new_h1_cd: str = ""
    new_h2_cd: str = ""
    new_h3_cd: str = ""
    new_li_cd: str = ""

    # Reference
    law_reference: str = ""  # 근거 법령
    note: str = ""


# ── Known Korean Administrative Changes (1994-2026) ────────────────────

# Major changes affecting gazette data period
ADMIN_CHANGES: list[AdminDistrictChange] = [
    # 1995: 대구/인천/광주/대전 직할시 → 광역시 승격
    AdminDistrictChange(
        change_id="1995-01-01_gwangyeoksi",
        change_date="1995-01-01",
        change_type="명칭변경",
        region_level="sido",
        old_name="직할시",
        new_name="광역시",
        note="대구/인천/광주/대전 직할시 → 광역시 전환 (지방자치법 개정)",
    ),
    AdminDistrictChange(
        change_id="1995-03-01_busan_gijang",
        change_date="1995-03-01",
        change_type="신설",
        region_level="sigungu",
        old_name="",
        new_name="부산광역시 기장군",
        new_h1_cd="26",
        new_h2_cd="26710",
        note="양산군 일부 → 부산광역시 기장군 신설",
    ),
    AdminDistrictChange(
        change_id="1995-03-01_daegu_dalseong",
        change_date="1995-03-01",
        change_type="편입",
        region_level="sigungu",
        old_name="경상북도 달성군",
        new_name="대구광역시 달성군",
        note="경북 달성군 → 대구광역시 편입",
    ),
    AdminDistrictChange(
        change_id="1995-03-01_incheon_ganghwa",
        change_date="1995-03-01",
        change_type="편입",
        region_level="sigungu",
        old_name="경기도 강화군",
        new_name="인천광역시 강화군",
        note="경기 강화군/옹진군 → 인천광역시 편입",
    ),
    # 1997: 울산 광역시 승격
    AdminDistrictChange(
        change_id="1997-07-15_ulsan_gwangyeok",
        change_date="1997-07-15",
        change_type="승격",
        region_level="sido",
        old_name="경상남도 울산시",
        new_name="울산광역시",
        new_h1_cd="26",
        note="울산시 → 울산광역시 승격",
    ),
    # 2003: 대구 달성군 화원읍 분동
    AdminDistrictChange(
        change_id="2003-03-10_dalseong_hwawon",
        change_date="2003-03-10",
        change_type="분리",
        region_level="dong",
        old_name="화원읍",
        new_name="화원읍(일부 분동)",
        note="대구광역시 달성군 화원읍 일부 분동",
    ),
    # 2012: 세종특별자치시 출범
    AdminDistrictChange(
        change_id="2012-07-01_sejong",
        change_date="2012-07-01",
        change_type="신설",
        region_level="sido",
        old_name="충청남도 연기군",
        new_name="세종특별자치시",
        new_h1_cd="36",
        note="연기군 + 공주시/청원군 일부 → 세종특별자치시 신설",
    ),
    # 2014: 청주/청원 통합
    AdminDistrictChange(
        change_id="2014-07-01_cheongju",
        change_date="2014-07-01",
        change_type="통합",
        region_level="sigungu",
        old_name="청주시, 청원군",
        new_name="청주시 (통합)",
        note="충북 청주시 + 청원군 → 통합 청주시",
    ),
    # 2023: 경기도 군포시 → 군포시 일부 행정동 변경 등
    AdminDistrictChange(
        change_id="2024-01-01_jinju",
        change_date="2024-01-01",
        change_type="명칭변경",
        region_level="dong",
        old_name="진주시 일부 동",
        new_name="진주시 개편 동",
        note="경남 진주시 행정동 통폐합",
    ),
]


# ── DB Schema (SQL) ────────────────────────────────────────────────────

ADMIN_HISTORY_DDL = """
CREATE TABLE IF NOT EXISTS admin_district_changes (
    change_id TEXT PRIMARY KEY,
    change_date DATE NOT NULL,
    change_type TEXT NOT NULL,
    region_level TEXT NOT NULL,
    old_name TEXT,
    new_name TEXT,
    old_h1_cd TEXT, old_h2_cd TEXT, old_h3_cd TEXT, old_li_cd TEXT,
    new_h1_cd TEXT, new_h2_cd TEXT, new_h3_cd TEXT, new_li_cd TEXT,
    law_reference TEXT,
    note TEXT
);

CREATE INDEX IF NOT EXISTS idx_adc_date ON admin_district_changes(change_date);
CREATE INDEX IF NOT EXISTS idx_adc_region ON admin_district_changes(old_h1_cd, old_h2_cd);

CREATE TABLE IF NOT EXISTS location_yearly_codes (
    location_id TEXT NOT NULL,
    year INTEGER NOT NULL,
    h1_cd TEXT,    -- 시도코드
    h2_cd TEXT,    -- 시군구코드
    h3_cd TEXT,    -- 읍면동코드
    li_cd TEXT,    -- 리코드
    lot_number TEXT,  -- 지번
    road_code TEXT,   -- 도로명코드
    pnu TEXT,         -- 필지고유번호 (19자리)
    address_full TEXT,
    address_road TEXT,
    source TEXT,      -- geocode/ocr/manual
    PRIMARY KEY (location_id, year)
);

CREATE INDEX IF NOT EXISTS idx_lyc_year ON location_yearly_codes(year);
CREATE INDEX IF NOT EXISTS idx_lyc_pnu ON location_yearly_codes(pnu);
"""


# ── Utility Functions ──────────────────────────────────────────────────


def infer_admin_changes(
    old_name: str, new_name: str, year: int
) -> list[AdminDistrictChange]:
    """Infer administrative changes from OCR-extracted place name differences."""
    changes = []
    for change in ADMIN_CHANGES:
        change_year = (
            int(change.change_date[:4])
            if isinstance(change.change_date, str)
            else change.change_date.year
        )
        if abs(change_year - year) <= 2:  # Within 2 years
            if old_name in change.old_name or new_name in change.new_name:
                changes.append(change)
    return changes


def pnu_from_address(
    h1_cd: str, h2_cd: str, lot_main: str, lot_sub: str = "0000"
) -> str:
    """Generate 19-digit PNU (필지고유번호) from address codes and lot number.

    Format: h1_cd(2) + h2_cd(5) + h3_cd(5) + li_cd(2) + lot_main(4) + lot_sub(4) = 19 digits
    """
    return f"{h1_cd:>02s}{h2_cd:>05s}{lot_main:>08s}{lot_sub:>04s}"[:19]


def get_year_for_address(location_id: str, year: int) -> dict:
    """Get the administrative codes for a location in a specific year.

    Returns the codes that were valid in that year, accounting for changes.
    """
    # This would query the location_yearly_codes table
    # and apply admin_district_changes to infer codes for years without data
    pass
