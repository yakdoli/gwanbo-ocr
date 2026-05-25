"""Korean Government Reference Data.

Comprehensive reference datasets derived from Korean government standards:
- Administrative district codes (행정구역코드) - 행정안전부
- Government organization codes (정부조직코드) - 행정안전부/법제처
- Legal reference codes (법령코드) - 법제처
- Gazette classification codes (관보분류코드)
- Stock market codes (증권시장코드) - 금융위원회/KRX
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as Date
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════
# 1. COMPREHENSIVE ADMINISTRATIVE DISTRICT CODES
# ═══════════════════════════════════════════════════════════════════════

# Full dong-level codes for Seoul (most referenced in gazette data)
# Format: h3_cd (8 digits), name_kr, h2_cd, valid_from, valid_to
DONG_CODES: list[dict] = [
    # 강남구 (11680)
    {
        "h3_cd": "11680101",
        "name_kr": "역삼1동",
        "h2_cd": "11680",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11680103",
        "name_kr": "역삼2동",
        "h2_cd": "11680",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11680105",
        "name_kr": "개포1동",
        "h2_cd": "11680",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11680106",
        "name_kr": "개포2동",
        "h2_cd": "11680",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11680108",
        "name_kr": "개포4동",
        "h2_cd": "11680",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11680118",
        "name_kr": "도곡1동",
        "h2_cd": "11680",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11680119",
        "name_kr": "도곡2동",
        "h2_cd": "11680",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11680110",
        "name_kr": "삼성1동",
        "h2_cd": "11680",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11680111",
        "name_kr": "삼성2동",
        "h2_cd": "11680",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11680121",
        "name_kr": "압구정동",
        "h2_cd": "11680",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11680122",
        "name_kr": "청담동",
        "h2_cd": "11680",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11680114",
        "name_kr": "대치1동",
        "h2_cd": "11680",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11680115",
        "name_kr": "대치2동",
        "h2_cd": "11680",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11680116",
        "name_kr": "대치4동",
        "h2_cd": "11680",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11680123",
        "name_kr": "세곡동",
        "h2_cd": "11680",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11680125",
        "name_kr": "자곡동",
        "h2_cd": "11680",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11680126",
        "name_kr": "율현동",
        "h2_cd": "11680",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11680124",
        "name_kr": "수서동",
        "h2_cd": "11680",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11680127",
        "name_kr": "일원1동",
        "h2_cd": "11680",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11680128",
        "name_kr": "일원본동",
        "h2_cd": "11680",
        "valid_from": "1989-01-01",
    },
    # 송파구 (11710)
    {
        "h3_cd": "11710101",
        "name_kr": "풍납1동",
        "h2_cd": "11710",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11710102",
        "name_kr": "풍납2동",
        "h2_cd": "11710",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11710103",
        "name_kr": "거여1동",
        "h2_cd": "11710",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11710104",
        "name_kr": "거여2동",
        "h2_cd": "11710",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11710105",
        "name_kr": "마천1동",
        "h2_cd": "11710",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11710106",
        "name_kr": "마천2동",
        "h2_cd": "11710",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11710107",
        "name_kr": "방이1동",
        "h2_cd": "11710",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11710108",
        "name_kr": "방이2동",
        "h2_cd": "11710",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11710109",
        "name_kr": "오금동",
        "h2_cd": "11710",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11710110",
        "name_kr": "송파1동",
        "h2_cd": "11710",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11710111",
        "name_kr": "송파2동",
        "h2_cd": "11710",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11710112",
        "name_kr": "석촌동",
        "h2_cd": "11710",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11710113",
        "name_kr": "삼전동",
        "h2_cd": "11710",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11710114",
        "name_kr": "가락1동",
        "h2_cd": "11710",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11710115",
        "name_kr": "가락2동",
        "h2_cd": "11710",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11710116",
        "name_kr": "가락본동",
        "h2_cd": "11710",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11710117",
        "name_kr": "문정1동",
        "h2_cd": "11710",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11710118",
        "name_kr": "문정2동",
        "h2_cd": "11710",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11710119",
        "name_kr": "장지동",
        "h2_cd": "11710",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11710120",
        "name_kr": "잠실2동",
        "h2_cd": "11710",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11710121",
        "name_kr": "잠실3동",
        "h2_cd": "11710",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11710122",
        "name_kr": "잠실4동",
        "h2_cd": "11710",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11710123",
        "name_kr": "잠실6동",
        "h2_cd": "11710",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11710124",
        "name_kr": "잠실7동",
        "h2_cd": "11710",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11710125",
        "name_kr": "잠실본동",
        "h2_cd": "11710",
        "valid_from": "1989-01-01",
    },
    # 서초구 (11650)
    {
        "h3_cd": "11650101",
        "name_kr": "서초1동",
        "h2_cd": "11650",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11650102",
        "name_kr": "서초2동",
        "h2_cd": "11650",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11650103",
        "name_kr": "서초3동",
        "h2_cd": "11650",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11650104",
        "name_kr": "서초4동",
        "h2_cd": "11650",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11650105",
        "name_kr": "잠원동",
        "h2_cd": "11650",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11650106",
        "name_kr": "반포1동",
        "h2_cd": "11650",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11650107",
        "name_kr": "반포2동",
        "h2_cd": "11650",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11650108",
        "name_kr": "반포3동",
        "h2_cd": "11650",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11650109",
        "name_kr": "반포4동",
        "h2_cd": "11650",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11650110",
        "name_kr": "반포본동",
        "h2_cd": "11650",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11650111",
        "name_kr": "방배1동",
        "h2_cd": "11650",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11650112",
        "name_kr": "방배2동",
        "h2_cd": "11650",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11650113",
        "name_kr": "방배3동",
        "h2_cd": "11650",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11650114",
        "name_kr": "방배4동",
        "h2_cd": "11650",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11650115",
        "name_kr": "방배본동",
        "h2_cd": "11650",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11650116",
        "name_kr": "양재1동",
        "h2_cd": "11650",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11650117",
        "name_kr": "양재2동",
        "h2_cd": "11650",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11650118",
        "name_kr": "내곡동",
        "h2_cd": "11650",
        "valid_from": "1989-01-01",
    },
    # 종로구 (11110)
    {
        "h3_cd": "11110101",
        "name_kr": "청운효자동",
        "h2_cd": "11110",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11110102",
        "name_kr": "사직동",
        "h2_cd": "11110",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11110103",
        "name_kr": "삼청동",
        "h2_cd": "11110",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11110104",
        "name_kr": "부암동",
        "h2_cd": "11110",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11110105",
        "name_kr": "평창동",
        "h2_cd": "11110",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11110106",
        "name_kr": "무악동",
        "h2_cd": "11110",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11110107",
        "name_kr": "교남동",
        "h2_cd": "11110",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11110108",
        "name_kr": "가회동",
        "h2_cd": "11110",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11110109",
        "name_kr": "종로1·2·3·4가동",
        "h2_cd": "11110",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11110110",
        "name_kr": "종로5·6가동",
        "h2_cd": "11110",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11110111",
        "name_kr": "이화동",
        "h2_cd": "11110",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11110112",
        "name_kr": "혜화동",
        "h2_cd": "11110",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11110113",
        "name_kr": "창신1동",
        "h2_cd": "11110",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11110114",
        "name_kr": "창신2동",
        "h2_cd": "11110",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11110115",
        "name_kr": "창신3동",
        "h2_cd": "11110",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11110116",
        "name_kr": "숭인1동",
        "h2_cd": "11110",
        "valid_from": "1989-01-01",
    },
    {
        "h3_cd": "11110117",
        "name_kr": "숭인2동",
        "h2_cd": "11110",
        "valid_from": "1989-01-01",
    },
]


# ═══════════════════════════════════════════════════════════════════════
# 2. GOVERNMENT ORGANIZATION REFERENCE
# ═══════════════════════════════════════════════════════════════════════

GOV_ORG_CODES: list[dict] = {
    "organizations": [
        # 중앙행정기관 (18부 4처 18청)
        {
            "org_code": "10001",
            "name_kr": "국토교통부",
            "name_alias": "건설교통부,국토해양부",
            "org_level": 1,
            "established": "1948-11-04",
            "parent": None,
            "law_ref": "정부조직법 제26조",
        },
        {
            "org_code": "10002",
            "name_kr": "기획재정부",
            "name_alias": "재정경제부,기획예산처",
            "org_level": 1,
            "established": "1948-11-04",
            "parent": None,
        },
        {
            "org_code": "10003",
            "name_kr": "행정안전부",
            "name_alias": "행정자치부,내무부,총무처,안전행정부",
            "org_level": 1,
            "established": "1948-11-04",
            "parent": None,
        },
        {
            "org_code": "10004",
            "name_kr": "농림축산식품부",
            "name_alias": "농림수산식품부,농림부,농수산부",
            "org_level": 1,
            "established": "1948-11-04",
            "parent": None,
        },
        {
            "org_code": "10005",
            "name_kr": "산업통상자원부",
            "name_alias": "지식경제부,산업자원부,통상산업부",
            "org_level": 1,
            "established": "1948-11-04",
            "parent": None,
        },
        {
            "org_code": "10006",
            "name_kr": "보건복지부",
            "name_alias": "보건사회부",
            "org_level": 1,
            "established": "1948-11-04",
            "parent": None,
        },
        {
            "org_code": "10007",
            "name_kr": "환경부",
            "name_alias": "",
            "org_level": 1,
            "established": "1980-01-01",
            "parent": None,
        },
        {
            "org_code": "10008",
            "name_kr": "고용노동부",
            "name_alias": "노동부",
            "org_level": 1,
            "established": "1948-11-04",
            "parent": None,
        },
        {
            "org_code": "10009",
            "name_kr": "여성가족부",
            "name_alias": "여성부",
            "org_level": 1,
            "established": "2001-01-29",
            "parent": None,
        },
        {
            "org_code": "10010",
            "name_kr": "국방부",
            "name_alias": "",
            "org_level": 1,
            "established": "1948-11-04",
            "parent": None,
        },
        {
            "org_code": "10011",
            "name_kr": "법무부",
            "name_alias": "",
            "org_level": 1,
            "established": "1948-11-04",
            "parent": None,
        },
        {
            "org_code": "10012",
            "name_kr": "외교부",
            "name_alias": "외교통상부,외무부",
            "org_level": 1,
            "established": "1948-11-04",
            "parent": None,
        },
        {
            "org_code": "10013",
            "name_kr": "교육부",
            "name_alias": "교육과학기술부,교육인적자원부,문교부",
            "org_level": 1,
            "established": "1948-11-04",
            "parent": None,
        },
        {
            "org_code": "10014",
            "name_kr": "문화체육관광부",
            "name_alias": "문화관광부,문화체육부,문화공보부",
            "org_level": 1,
            "established": "1948-11-04",
            "parent": None,
        },
        {
            "org_code": "10015",
            "name_kr": "과학기술정보통신부",
            "name_alias": "미래창조과학부,정보통신부,체신부",
            "org_level": 1,
            "established": "2017-07-26",
            "parent": None,
        },
        {
            "org_code": "10016",
            "name_kr": "해양수산부",
            "name_alias": "",
            "org_level": 1,
            "established": "2013-03-23",
            "parent": None,
        },
        {
            "org_code": "10017",
            "name_kr": "중소벤처기업부",
            "name_alias": "중소기업청",
            "org_level": 1,
            "established": "2017-07-26",
            "parent": None,
        },
        # 대통령 직속 기관
        {
            "org_code": "20001",
            "name_kr": "감사원",
            "name_alias": "",
            "org_level": 1,
            "established": "1948-11-04",
            "parent": None,
        },
        {
            "org_code": "20002",
            "name_kr": "국가정보원",
            "name_alias": "",
            "org_level": 1,
            "established": "1961-06-10",
            "parent": None,
        },
        {
            "org_code": "20003",
            "name_kr": "방송통신위원회",
            "name_alias": "",
            "org_level": 1,
            "established": "2008-02-29",
            "parent": None,
        },
        {
            "org_code": "20004",
            "name_kr": "공정거래위원회",
            "name_alias": "",
            "org_level": 1,
            "established": "1981-04-01",
            "parent": None,
        },
        {
            "org_code": "20005",
            "name_kr": "금융위원회",
            "name_alias": "금융감독원",
            "org_level": 1,
            "established": "2008-02-29",
            "parent": None,
        },
        {
            "org_code": "20006",
            "name_kr": "국민권익위원회",
            "name_alias": "",
            "org_level": 1,
            "established": "2008-02-29",
            "parent": None,
        },
        {
            "org_code": "20007",
            "name_kr": "국가인권위원회",
            "name_alias": "",
            "org_level": 1,
            "established": "2001-11-25",
            "parent": None,
        },
        # 국무총리 직속 기관
        {
            "org_code": "30001",
            "name_kr": "국세청",
            "name_alias": "관세청",
            "org_level": 2,
            "established": "1966-03-03",
            "parent": "10002",
        },
        {
            "org_code": "30002",
            "name_kr": "경찰청",
            "name_alias": "",
            "org_level": 2,
            "established": "1991-07-31",
            "parent": "10003",
        },
        {
            "org_code": "30003",
            "name_kr": "특허청",
            "name_alias": "",
            "org_level": 2,
            "established": "1977-03-12",
            "parent": "10005",
        },
        {
            "org_code": "30004",
            "name_kr": "기상청",
            "name_alias": "",
            "org_level": 2,
            "established": "1990-12-27",
            "parent": None,
        },
        {
            "org_code": "30005",
            "name_kr": "통계청",
            "name_alias": "",
            "org_level": 2,
            "established": "1990-12-27",
            "parent": None,
        },
        {
            "org_code": "30006",
            "name_kr": "문화재청",
            "name_alias": "",
            "org_level": 2,
            "established": "1999-05-24",
            "parent": "10014",
        },
        {
            "org_code": "30007",
            "name_kr": "산림청",
            "name_alias": "",
            "org_level": 2,
            "established": "1967-01-01",
            "parent": "10004",
        },
        {
            "org_code": "30008",
            "name_kr": "농촌진흥청",
            "name_alias": "",
            "org_level": 2,
            "established": "1962-04-01",
            "parent": "10004",
        },
        {
            "org_code": "30009",
            "name_kr": "방위사업청",
            "name_alias": "",
            "org_level": 2,
            "established": "2006-01-01",
            "parent": "10010",
        },
        {
            "org_code": "30010",
            "name_kr": "인사혁신처",
            "name_alias": "중앙인사위원회",
            "org_level": 2,
            "established": "2014-11-19",
            "parent": None,
        },
        # 사법/입법/헌법 기관
        {
            "org_code": "40001",
            "name_kr": "대법원",
            "name_alias": "",
            "org_level": 1,
            "established": "1948-11-04",
            "parent": None,
        },
        {
            "org_code": "40002",
            "name_kr": "헌법재판소",
            "name_alias": "",
            "org_level": 1,
            "established": "1988-09-01",
            "parent": None,
        },
        {
            "org_code": "40003",
            "name_kr": "중앙선거관리위원회",
            "name_alias": "",
            "org_level": 1,
            "established": "1948-11-04",
            "parent": None,
        },
        {
            "org_code": "40004",
            "name_kr": "법원행정처",
            "name_alias": "",
            "org_level": 2,
            "established": "1948-11-04",
            "parent": "40001",
        },
    ],
}


# ═══════════════════════════════════════════════════════════════════════
# 3. LEGAL/GAZETTE REFERENCE CODES
# ═══════════════════════════════════════════════════════════════════════

LEGAL_CODES = {
    "document_types": [
        {
            "code": "DT01",
            "name_kr": "법률",
            "name_en": "Act",
            "hierarchy": 1,
            "issuing_body": "국회",
        },
        {
            "code": "DT02",
            "name_kr": "대통령령",
            "name_en": "Presidential Decree",
            "hierarchy": 2,
            "issuing_body": "대통령",
        },
        {
            "code": "DT03",
            "name_kr": "국무총리령",
            "name_en": "Prime Ministerial Decree",
            "hierarchy": 3,
            "issuing_body": "국무총리",
        },
        {
            "code": "DT04",
            "name_kr": "부령",
            "name_en": "Ministerial Decree",
            "hierarchy": 4,
            "issuing_body": "각 부처",
        },
        {
            "code": "DT05",
            "name_kr": "고시",
            "name_en": "Public Notification",
            "hierarchy": 5,
            "issuing_body": "각급 행정기관",
        },
        {
            "code": "DT06",
            "name_kr": "공고",
            "name_en": "Public Announcement",
            "hierarchy": 5,
            "issuing_body": "각급 행정기관",
        },
        {
            "code": "DT07",
            "name_kr": "훈령",
            "name_en": "Directive",
            "hierarchy": 6,
            "issuing_body": "상급기관",
        },
        {
            "code": "DT08",
            "name_kr": "예규",
            "name_en": "Established Rule",
            "hierarchy": 6,
            "issuing_body": "각급 행정기관",
        },
        {
            "code": "DT09",
            "name_kr": "지침",
            "name_en": "Guideline",
            "hierarchy": 7,
            "issuing_body": "각급 행정기관",
        },
        {
            "code": "DT10",
            "name_kr": "시행령",
            "name_en": "Enforcement Decree",
            "hierarchy": 3,
            "issuing_body": "대통령",
        },
        {
            "code": "DT11",
            "name_kr": "시행규칙",
            "name_en": "Enforcement Rule",
            "hierarchy": 4,
            "issuing_body": "각 부처",
        },
    ],
    "notice_types": [
        {"code": "NT01", "name_kr": "제정", "name_en": "Enactment"},
        {"code": "NT02", "name_kr": "일부개정", "name_en": "Partial Amendment"},
        {"code": "NT03", "name_kr": "전부개정", "name_en": "Full Amendment"},
        {"code": "NT04", "name_kr": "폐지", "name_en": "Abolition"},
        {"code": "NT05", "name_kr": "타법개정", "name_en": "Revision by Other Act"},
        {"code": "NT06", "name_kr": "신설", "name_en": "New Establishment"},
    ],
    "gazette_categories": [
        {"code": "GC01", "name_kr": "헌법", "name_en": "Constitution"},
        {"code": "GC02", "name_kr": "법률", "name_en": "Laws"},
        {"code": "GC03", "name_kr": "조약", "name_en": "Treaties"},
        {"code": "GC04", "name_kr": "대통령령", "name_en": "Presidential Decrees"},
        {"code": "GC05", "name_kr": "총리령·부령", "name_en": "Ministerial Decrees"},
        {
            "code": "GC06",
            "name_kr": "훈령·예규·고시",
            "name_en": "Directives, Rules, Notices",
        },
        {"code": "GC07", "name_kr": "공고", "name_en": "Announcements"},
        {"code": "GC08", "name_kr": "국회", "name_en": "National Assembly"},
        {"code": "GC09", "name_kr": "법원", "name_en": "Courts"},
        {"code": "GC10", "name_kr": "헌법재판소", "name_en": "Constitutional Court"},
        {"code": "GC11", "name_kr": "선거관리위원회", "name_en": "Election Commission"},
        {"code": "GC12", "name_kr": "감사원", "name_en": "Board of Audit"},
        {"code": "GC13", "name_kr": "지방자치단체", "name_en": "Local Government"},
        {"code": "GC14", "name_kr": "인사", "name_en": "Personnel"},
        {"code": "GC15", "name_kr": "기타", "name_en": "Others"},
    ],
}


# ═══════════════════════════════════════════════════════════════════════
# 4. FINANCIAL REFERENCE CODES
# ═══════════════════════════════════════════════════════════════════════

FINANCIAL_CODES = {
    "stock_exchanges": [
        {
            "code": "KRX",
            "name_kr": "한국거래소",
            "name_en": "Korea Exchange",
            "markets": ["KOSPI", "KOSDAQ", "KONEX"],
        },
    ],
    "markets": [
        {
            "code": "KOSPI",
            "name_kr": "유가증권시장",
            "name_en": "KOSPI Market",
            "established": "1956-03-03",
        },
        {
            "code": "KOSDAQ",
            "name_kr": "코스닥시장",
            "name_en": "KOSDAQ Market",
            "established": "1996-07-01",
        },
        {
            "code": "KONEX",
            "name_kr": "코넥스시장",
            "name_en": "KONEX Market",
            "established": "2013-07-01",
        },
        {"code": "OTC", "name_kr": "장외시장", "name_en": "OTC Market"},
        {"code": "NON_LISTED", "name_kr": "비상장", "name_en": "Unlisted"},
    ],
    "currency_units": [
        {"code": "WON", "name_kr": "원", "name_en": "KRW", "is_base": True},
        {"code": "THOUSAND_WON", "name_kr": "천원", "multiplier": 1000},
        {"code": "TEN_THOUSAND_WON", "name_kr": "만원", "multiplier": 10000},
        {"code": "MILLION_WON", "name_kr": "백만원", "multiplier": 1000000},
        {"code": "HUNDRED_MILLION_WON", "name_kr": "억원", "multiplier": 100000000},
    ],
    "real_estate_units": [
        {"code": "M2", "name_kr": "㎡", "name_en": "Square Meter"},
        {"code": "PYEONG", "name_kr": "평", "name_en": "Pyeong", "to_m2": 3.305785},
        {"code": "HA", "name_kr": "ha", "name_en": "Hectare", "to_m2": 10000},
    ],
    "vehicle_types": [
        {"code": "SEDAN", "name_kr": "승용", "name_en": "Sedan"},
        {"code": "SUV", "name_kr": "승용(SUV)", "name_en": "SUV"},
        {"code": "VAN", "name_kr": "승합", "name_en": "Van"},
        {"code": "TRUCK", "name_kr": "화물", "name_en": "Truck"},
    ],
    "asset_classes": [
        {"code": "REAL_ESTATE", "name_kr": "부동산", "name_en": "Real Estate"},
        {"code": "LISTED_STOCK", "name_kr": "상장주식", "name_en": "Listed Stock"},
        {
            "code": "UNLISTED_STOCK",
            "name_kr": "비상장주식",
            "name_en": "Unlisted Stock",
        },
        {"code": "BOND", "name_kr": "채권", "name_en": "Bond"},
        {"code": "CASH_DEPOSIT", "name_kr": "예금", "name_en": "Cash/Deposit"},
        {"code": "MEMBERSHIP", "name_kr": "회원권", "name_en": "Membership"},
        {"code": "VEHICLE", "name_kr": "자동차", "name_en": "Vehicle"},
        {"code": "GOLD", "name_kr": "금", "name_en": "Gold"},
        {"code": "OTHER", "name_kr": "기타", "name_en": "Other"},
    ],
}


# ═══════════════════════════════════════════════════════════════════════
# 5. REFERENCE DATA DDL
# ═══════════════════════════════════════════════════════════════════════

REFERENCE_DDL = """
-- Administrative codes
CREATE TABLE IF NOT EXISTS dong_codes (
    h3_cd TEXT PRIMARY KEY,
    name_kr TEXT NOT NULL,
    h2_cd TEXT,
    valid_from DATE,
    valid_to DATE
);

-- Government organizations
CREATE TABLE IF NOT EXISTS gov_org_codes (
    org_code TEXT PRIMARY KEY,
    name_kr TEXT NOT NULL,
    name_alias TEXT,
    org_level INTEGER,
    established DATE,
    parent_code TEXT
);

-- Document types
CREATE TABLE IF NOT EXISTS document_type_codes (
    code TEXT PRIMARY KEY,
    name_kr TEXT NOT NULL,
    name_en TEXT,
    hierarchy INTEGER,
    issuing_body TEXT
);

-- Notice types
CREATE TABLE IF NOT EXISTS notice_type_codes (
    code TEXT PRIMARY KEY,
    name_kr TEXT NOT NULL,
    name_en TEXT
);

-- Gazette categories
CREATE TABLE IF NOT EXISTS gazette_category_codes (
    code TEXT PRIMARY KEY,
    name_kr TEXT NOT NULL,
    name_en TEXT
);

-- Financial reference
CREATE TABLE IF NOT EXISTS stock_markets (
    code TEXT PRIMARY KEY,
    name_kr TEXT NOT NULL,
    name_en TEXT,
    established TEXT
);

CREATE TABLE IF NOT EXISTS currency_units (
    code TEXT PRIMARY KEY,
    name_kr TEXT NOT NULL,
    multiplier INTEGER
);

CREATE TABLE IF NOT EXISTS asset_class_codes (
    code TEXT PRIMARY KEY,
    name_kr TEXT NOT NULL,
    name_en TEXT
);

CREATE TABLE IF NOT EXISTS real_estate_units (
    code TEXT PRIMARY KEY,
    name_kr TEXT NOT NULL,
    name_en TEXT,
    to_m2 DOUBLE
);
"""


def load_gov_reference_data(db_path: str = "data/gwanbo.db") -> dict:
    """Load all government reference data into DuckDB."""
    import duckdb

    conn = duckdb.connect(db_path)

    # Create tables
    for stmt in REFERENCE_DDL.split(";"):
        stmt = stmt.strip()
        if stmt and not stmt.startswith("--"):
            try:
                conn.execute(stmt)
            except:
                pass

    loaded = {}

    # Dong codes
    for d in DONG_CODES:
        conn.execute(
            "INSERT OR IGNORE INTO dong_codes (h3_cd,name_kr,h2_cd,valid_from,valid_to) VALUES (?,?,?,?,?)",
            [d["h3_cd"], d["name_kr"], d["h2_cd"], d["valid_from"], d.get("valid_to")],
        )
    loaded["dong_codes"] = len(DONG_CODES)

    # Gov org codes
    for org in GOV_ORG_CODES["organizations"]:
        conn.execute(
            "INSERT OR IGNORE INTO gov_org_codes (org_code,name_kr,name_alias,org_level,established,parent_code) VALUES (?,?,?,?,?,?)",
            [
                org["org_code"],
                org["name_kr"],
                org["name_alias"],
                org["org_level"],
                org["established"],
                org.get("parent"),
            ],
        )
    loaded["gov_org_codes"] = len(GOV_ORG_CODES["organizations"])

    # Document types
    for dt in LEGAL_CODES["document_types"]:
        conn.execute(
            "INSERT OR IGNORE INTO document_type_codes (code,name_kr,name_en,hierarchy,issuing_body) VALUES (?,?,?,?,?)",
            [
                dt["code"],
                dt["name_kr"],
                dt["name_en"],
                dt["hierarchy"],
                dt["issuing_body"],
            ],
        )
    loaded["document_type_codes"] = len(LEGAL_CODES["document_types"])

    # Notice types
    for nt in LEGAL_CODES["notice_types"]:
        conn.execute(
            "INSERT OR IGNORE INTO notice_type_codes (code,name_kr,name_en) VALUES (?,?,?)",
            [nt["code"], nt["name_kr"], nt["name_en"]],
        )
    loaded["notice_type_codes"] = len(LEGAL_CODES["notice_types"])

    # Gazette categories
    for gc in LEGAL_CODES["gazette_categories"]:
        conn.execute(
            "INSERT OR IGNORE INTO gazette_category_codes (code,name_kr,name_en) VALUES (?,?,?)",
            [gc["code"], gc["name_kr"], gc["name_en"]],
        )
    loaded["gazette_category_codes"] = len(LEGAL_CODES["gazette_categories"])

    # Stock markets
    for sm in FINANCIAL_CODES["markets"]:
        conn.execute(
            "INSERT OR IGNORE INTO stock_markets (code,name_kr,name_en,established) VALUES (?,?,?,?)",
            [sm["code"], sm["name_kr"], sm["name_en"], sm.get("established")],
        )
    loaded["stock_markets"] = len(FINANCIAL_CODES["markets"])

    # Currency units
    for cu in FINANCIAL_CODES["currency_units"]:
        conn.execute(
            "INSERT OR IGNORE INTO currency_units (code,name_kr,multiplier) VALUES (?,?,?)",
            [cu["code"], cu["name_kr"], cu.get("multiplier", 1)],
        )
    loaded["currency_units"] = len(FINANCIAL_CODES["currency_units"])

    # Asset classes
    for ac in FINANCIAL_CODES["asset_classes"]:
        conn.execute(
            "INSERT OR IGNORE INTO asset_class_codes (code,name_kr,name_en) VALUES (?,?,?)",
            [ac["code"], ac["name_kr"], ac["name_en"]],
        )
    loaded["asset_class_codes"] = len(FINANCIAL_CODES["asset_classes"])

    # Real estate units
    for ru in FINANCIAL_CODES["real_estate_units"]:
        conn.execute(
            "INSERT OR IGNORE INTO real_estate_units (code,name_kr,name_en,to_m2) VALUES (?,?,?,?)",
            [ru["code"], ru["name_kr"], ru["name_en"], ru.get("to_m2")],
        )
    loaded["real_estate_units"] = len(FINANCIAL_CODES["real_estate_units"])

    conn.commit()
    conn.close()
    return loaded
