# config.py
import os

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass  # python-dotenv not installed; rely on environment variables being set externally

# The adopted genome, applied once here because every entry point imports this
# module. .env is loaded first on purpose: a value there wins, exactly like a
# shell value, so a one-off experiment does not need a revert.
try:
    from core import adopted as _adopted

    _ADOPTED = _adopted.load()
    if _ADOPTED:
        _adopted.apply(_ADOPTED.get("genome") or {})
except Exception as _exc:
    # A broken adoption must never stop a run, but it must never be silent
    # either: a hand-edited value of the wrong type would otherwise leave a
    # tens-of-hours retrain running on production defaults with no adoption in
    # force and nothing on screen to say so.
    print(f"[adopt] adopted genome ignored: {_exc}")
    _ADOPTED = None

# --- 1. MODEL PARAMETERS ---
SEQ_LEN = 10

# Buy/sell thresholds per asset
THRESHOLDS = {
    "DEFAULT": 0.55,    # Base threshold

    # ELITE (lower threshold for high-performing assets)
    "TSLA": 0.53,
    "ETH": 0.53,
    "GOLD": 0.54,
    "VIX": 0.53,

    # CAUTIOUS (higher threshold for safety)
    "TON": 0.60,
    "NASDAQ": 0.58,
    "SOL": 0.58,
    "DXY": 0.58,

    # CONSERVATIVE
    "BTC": 0.54,
    "SBER": 0.54
}

# Telegram settings - loaded from .env (never hardcode credentials)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_USER_ID = os.getenv("TELEGRAM_USER_ID", "")

# Proxy: empty string = no proxy. The address is set only via .env.
SOCKS5_PROXY = os.getenv("SOCKS5_PROXY", "")

# --- 2. ASSET MAP ---
FULL_ASSET_MAP = {
    # GLOBAL INDICES
    'VIX': '^VIX', 'DXY': 'DX-Y.NYB', 'TNX': '^TNX',
    'SP500': '^GSPC', 'NASDAQ': '^IXIC', 'DOW': '^DJI', 'RUSSELL': '^RUT',

    # WORLD INDICES
    'NIKKEI': '^N225', 'HANGSENG': '^HSI', 'KOSPI': '^KS11', 'ASX200': '^AXJO',
    'TSX': '^GSPTSE', 'BOVESPA': '^BVSP', 'NIFTY': '^NSEI', 'SHANGHAI': '000001.SS',

    # COMMODITIES
    'GOLD': 'GC=F', 'SILVER': 'SI=F', 'OIL': 'CL=F', 'GAS': 'NG=F',
    'BRENT': 'BZ=F', 'COPPER': 'HG=F', 'PLATINUM': 'PL=F', 'PALLADIUM': 'PA=F',
    'WHEAT': 'ZW=F', 'CORN': 'ZC=F', 'SOYBEAN': 'ZS=F',
    'COFFEE': 'KC=F', 'SUGAR': 'SB=F', 'COCOA': 'CC=F',

    # US TECH
    'NVDA': 'NVDA', 'TSLA': 'TSLA', 'AAPL': 'AAPL', 'MSFT': 'MSFT',
    'GOOGL': 'GOOGL', 'AMZN': 'AMZN', 'META': 'META', 'AMD': 'AMD',
    'PLTR': 'PLTR', 'COIN': 'COIN', 'MSTR': 'MSTR',
    # AI INFRASTRUCTURE
    'CRWV': 'CRWV', 'NBIS': 'NBIS', 'ARM': 'ARM', 'ANET': 'ANET', 'DELL': 'DELL',
    # OPENAI / ANTHROPIC EXPOSURE. Both are PRIVATE companies - there is no share to
    # quote, so these are the listed proxies: DXYZ is a closed-end fund holding an
    # OpenAI stake, SOFTBANK is OpenAI's largest listed shareholder, ARKVX is an
    # interval fund holding BOTH OpenAI and Anthropic (NAV-priced and thin - treat
    # its signals with care). MSFT / AMZN / GOOGL above are the funders themselves.
    'DXYZ': 'DXYZ', 'SOFTBANK': 'SFTBY', 'ARKVX': 'ARKVX',
    # ASIA / LATAM TECH
    'TENCENT': 'TCEHY', 'SAMSUNG': '005930.KS', 'BYD': 'BYDDY', 'INFOSYS': 'INFY',
    'PDD': 'PDD', 'JD': 'JD', 'NIO': 'NIO', 'XPEV': 'XPEV', 'SE': 'SE', 'MELI': 'MELI',

    # US HEALTHCARE
    'JNJ': 'JNJ', 'UNH': 'UNH', 'PFE': 'PFE', 'LLY': 'LLY',
    'ABBV': 'ABBV', 'MRK': 'MRK',
    'TMO': 'TMO', 'ABT': 'ABT', 'AMGN': 'AMGN', 'GILD': 'GILD',
    'BMY': 'BMY', 'ISRG': 'ISRG', 'VRTX': 'VRTX',

    # US FINANCE
    'JPM': 'JPM', 'BAC': 'BAC', 'GS': 'GS', 'V': 'V',
    'MA': 'MA', 'WFC': 'WFC',
    'MS': 'MS', 'CITI': 'C', 'BLK': 'BLK', 'SCHW': 'SCHW',
    'HOOD': 'HOOD', 'SPGI': 'SPGI', 'HDFC': 'HDB',

    # US CONSUMER
    'WMT': 'WMT', 'KO': 'KO', 'PEP': 'PEP', 'MCD': 'MCD',
    'NKE': 'NKE', 'DIS': 'DIS', 'NFLX': 'NFLX', 'SBUX': 'SBUX',
    'ABNB': 'ABNB', 'BKNG': 'BKNG', 'LULU': 'LULU', 'CMG': 'CMG',
    'TGT': 'TGT', 'LOW': 'LOW', 'MO': 'MO', 'MDLZ': 'MDLZ',

    # US INDUSTRIAL & ENERGY
    'BA': 'BA', 'CAT': 'CAT', 'XOM': 'XOM', 'CVX': 'CVX', 'COP': 'COP',
    'GE': 'GE', 'HON': 'HON', 'LMT': 'LMT', 'RTX': 'RTX', 'UNP': 'UNP',
    'DE': 'DE', 'SLB': 'SLB', 'OXY': 'OXY',
    # DATA-CENTER POWER - the electricity side of the AI build-out
    'VRT': 'VRT', 'CEG': 'CEG', 'VST': 'VST', 'TLN': 'TLN',
    'NEE': 'NEE', 'OKLO': 'OKLO', 'GEV': 'GEV',

    # US SEMICONDUCTORS
    'INTC': 'INTC', 'QCOM': 'QCOM', 'AVGO': 'AVGO', 'MU': 'MU',
    'MRVL': 'MRVL', 'SMCI': 'SMCI', 'MPWR': 'MPWR', 'TXN': 'TXN', 'ADI': 'ADI',
    'LRCX': 'LRCX', 'AMAT': 'AMAT', 'KLAC': 'KLAC', 'NXPI': 'NXPI',
    'ONSEMI': 'ON', 'GFS': 'GFS',

    # US SOFTWARE
    'CRM': 'CRM', 'ORCL': 'ORCL', 'ADBE': 'ADBE', 'UBER': 'UBER', 'PYPL': 'PYPL',
    'NOW': 'NOW', 'SNOW': 'SNOW', 'DDOG': 'DDOG', 'MDB': 'MDB', 'NET': 'NET',
    'PANW': 'PANW', 'CRWD': 'CRWD', 'ZS': 'ZS', 'TEAM': 'TEAM', 'SHOP': 'SHOP',
    'INTU': 'INTU', 'WDAY': 'WDAY',

    # EU INDICES (native exchange tickers on Yahoo)
    'DAX': '^GDAXI', 'CAC40': '^FCHI', 'ESTOXX50': '^STOXX50E', 'FTSE100': '^FTSE',
    'IBEX35': '^IBEX', 'FTSEMIB': 'FTSEMIB.MI', 'AEX': '^AEX', 'SMI': '^SSMI',

    # EU STOCKS - top European large caps (key is the native EU listing on Yahoo)
    'ASML': 'ASML.AS', 'LVMH': 'MC.PA', 'SAP': 'SAP.DE', 'NESTLE': 'NESN.SW',
    'NOVO': 'NOVO-B.CO', 'AZN': 'AZN.L', 'SHELL': 'SHEL.L', 'TOTAL': 'TTE.PA',
    'SIEMENS': 'SIE.DE', 'AIRBUS': 'AIR.PA', 'LOREAL': 'OR.PA', 'ALLIANZ': 'ALV.DE',
    'HERMES': 'RMS.PA', 'SCHNEIDER': 'SU.PA', 'SANTANDER': 'SAN.MC', 'BNP': 'BNP.PA',
    'ENEL': 'ENEL.MI', 'IBERDROLA': 'IBE.MC',

    # WELL-KNOWN GLOBAL BRANDS - US-listed (incl. ADRs)
    'COST': 'COST', 'HD': 'HD', 'PG': 'PG', 'COLG': 'CL', 'ESTEE': 'EL',
    'FORD': 'F', 'GM': 'GM', 'FERRARI': 'RACE', 'STELLANTIS': 'STLA',
    'TOYOTA': 'TM', 'SPOTIFY': 'SPOT', 'AMEX': 'AXP', 'IBM': 'IBM',
    'CISCO': 'CSCO', 'SONY': 'SONY', 'TSMC': 'TSM', 'ALIBABA': 'BABA',
    # WELL-KNOWN GLOBAL BRANDS - EU-listed
    'VW': 'VOW3.DE', 'MERCEDES': 'MBG.DE', 'BMW': 'BMW.DE', 'PORSCHE': 'P911.DE',
    'ADIDAS': 'ADS.DE', 'KERING': 'KER.PA', 'RICHEMONT': 'CFR.SW',
    'UNILEVER': 'ULVR.L', 'DIAGEO': 'DGE.L', 'ABINBEV': 'ABI.BR',

    # CRYPTO
    'BTC': 'BTC-USD', 'ETH': 'ETH-USD', 'SOL': 'SOL-USD',
    'XRP': 'XRP-USD', 'TON': 'TON11419-USD', 'DOGE': 'DOGE-USD', 'BNB': 'BNB-USD',
    'ADA': 'ADA-USD', 'AVAX': 'AVAX-USD', 'DOT': 'DOT-USD', 'LINK': 'LINK-USD',
    'SHIB': 'SHIB-USD', 'ATOM': 'ATOM-USD', 'UNI': 'UNI7083-USD', 'NEAR': 'NEAR-USD',  # UNI-USD is empty on Yahoo, Uniswap trades under UNI7083-USD
    'LTC': 'LTC-USD', 'BCH': 'BCH-USD', 'TRX': 'TRX-USD', 'XLM': 'XLM-USD',
    'ETC': 'ETC-USD', 'HBAR': 'HBAR-USD', 'ICP': 'ICP-USD', 'FIL': 'FIL-USD',
    'AAVE': 'AAVE-USD', 'POL': 'POL-USD', 'OP': 'OP-USD',
    # Yahoo disambiguates these with a numeric suffix, the bare symbol returns nothing
    'APT': 'APT21794-USD', 'ARB': 'ARB11841-USD', 'SUI': 'SUI20947-USD',
    'PEPE': 'PEPE24478-USD', 'TAO': 'TAO22974-USD',
    # AI-themed tokens. WLD (Worldcoin, Sam Altman's project) is the one asset that
    # reprices on OpenAI news 24/7, which the equity proxies above cannot do.
    'FET': 'FET-USD', 'RENDER': 'RENDER-USD', 'WLD': 'WLD-USD',

    # RU MARKET - Blue chips
    'IMOEX': 'IMOEX', 'SBER': 'SBER', 'GAZP': 'GAZP', 'LKOH': 'LKOH',
    'ROSN': 'ROSN', 'NVTK': 'NVTK', 'TATN': 'TATN', 'SNGS': 'SNGS',
    'PLZL': 'PLZL', 'SIBN': 'SIBN', 'MGNT': 'MGNT',
    # RU - Banks and finance
    'TCSG': 'T', 'VTBR': 'VTBR', 'BSPB': 'BSPB', 'MOEX_EX': 'MOEX',
    # RU - Tech and growth
    'YNDX': 'YDEX', 'OZON': 'OZON', 'VKCO': 'VKCO', 'POSI': 'POSI',
    'MTSS': 'MTSS', 'RTKM': 'RTKM',
    # RU - Metals and industry
    'CHMF': 'CHMF', 'NLMK': 'NLMK', 'MAGN': 'MAGN',
    'RUAL': 'RUAL', 'ALRS': 'ALRS',
    # RU - Energy and transport
    'IRAO': 'IRAO', 'HYDR': 'HYDR', 'FLOT': 'FLOT',
    'AFLT': 'AFLT', 'PIKK': 'PIKK',
    # RU - Chemicals and fertilizers
    'PHOR': 'PHOR', 'SGZH': 'SGZH',
    # RU - Retail
    'FIVE': 'X5', 'FIXP': 'FIXR', 'LENT': 'LENT', 'MVID': 'MVID',  # FIVE and FIXP renamed on MOEX to X5 and FIXR
    # RU - Construction and development
    'SMLT': 'SMLT', 'LSRG': 'LSRG',
    # RU - Banks and finance (extra)
    'CBOM': 'CBOM',
    # RU - Energy (extra)
    'FEES': 'FEES', 'UPRO': 'UPRO', 'MSNG': 'MSNG',
    # RU - Industry (extra)
    'TRMK': 'TRMK', 'MTLR': 'MTLR', 'RASP': 'RASP', 'NMTP': 'NMTP',
    # RU - IT (extra)
    'HHRU': 'HEAD', 'SOFL': 'SOFL', 'ASTR': 'ASTR', 'WUSH': 'WUSH',  # HHRU renamed on MOEX to HEAD

    # FOREX - Majors
    'EURUSD': 'EURUSD=X', 'GBPUSD': 'GBPUSD=X', 'USDJPY': 'JPY=X',
    'USDCHF': 'CHF=X', 'AUDUSD': 'AUDUSD=X', 'USDCAD': 'CAD=X',
    'NZDUSD': 'NZDUSD=X', 'USDRUB': 'RUB=X',
    # FOREX - EUR crosses
    'EURGBP': 'EURGBP=X', 'EURJPY': 'EURJPY=X', 'EURCHF': 'EURCHF=X',
    'EURAUD': 'EURAUD=X', 'EURCAD': 'EURCAD=X', 'EURNZD': 'EURNZD=X',
    # FOREX - GBP crosses
    'GBPJPY': 'GBPJPY=X', 'GBPAUD': 'GBPAUD=X', 'GBPCAD': 'GBPCAD=X',
    'GBPCHF': 'GBPCHF=X', 'GBPNZD': 'GBPNZD=X',
    # FOREX - AUD/NZD/CAD/CHF crosses
    'AUDCAD': 'AUDCAD=X', 'AUDCHF': 'AUDCHF=X', 'AUDJPY': 'AUDJPY=X',
    'AUDNZD': 'AUDNZD=X', 'CADJPY': 'CADJPY=X', 'CHFJPY': 'CHFJPY=X',
    'NZDJPY': 'NZDJPY=X',
    # FOREX - Exotics
    'USDTRY': 'TRY=X', 'USDMXN': 'MXN=X', 'USDZAR': 'ZAR=X',
    'USDSGD': 'SGD=X', 'USDNOK': 'NOK=X', 'USDSEK': 'SEK=X',
    'USDPLN': 'PLN=X', 'USDCNH': 'CNY=X',  # CNH=X is empty on Yahoo (1 bar), CNY=X returns full history; offshore/onshore yuan differ by pips
}

# --- 3. GROUPING ---
ASSET_TYPES = {
    "TOP SIGNALS": ["ETH", "TSLA", "GOLD", "VIX", "PLTR", "IMOEX"],
    "CRYPTO": ["BTC", "ETH", "SOL", "XRP", "TON", "DOGE", "BNB",
               "ADA", "AVAX", "DOT", "LINK", "SHIB", "ATOM", "UNI", "NEAR",
               "LTC", "BCH", "TRX", "XLM", "ETC", "HBAR", "ICP", "FIL", "AAVE",
               "POL", "OP", "APT", "ARB", "SUI", "PEPE",
               "TAO", "FET", "RENDER", "WLD"],
    "COMMODITIES": ["GOLD", "SILVER", "OIL", "GAS", "BRENT", "COPPER", "PLATINUM",
                    "PALLADIUM", "WHEAT", "CORN", "SOYBEAN", "COFFEE", "SUGAR", "COCOA"],
    "INDICES & MACRO": ["SP500", "NASDAQ", "DOW", "RUSSELL", "IMOEX", "VIX", "DXY", "TNX",
                        "NIKKEI", "HANGSENG", "KOSPI", "ASX200", "TSX", "BOVESPA",
                        "NIFTY", "SHANGHAI"],
    "US TECH": ["NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "AMD", "PLTR", "COIN", "MSTR", "SONY", "TSMC", "ALIBABA",
                "CRWV", "NBIS", "ARM", "ANET", "DELL",
                "DXYZ", "SOFTBANK", "ARKVX",
                "TENCENT", "SAMSUNG", "BYD", "INFOSYS", "PDD", "JD", "NIO", "XPEV", "SE", "MELI"],
    "US HEALTHCARE": ["JNJ", "UNH", "PFE", "LLY", "ABBV", "MRK",
                      "TMO", "ABT", "AMGN", "GILD", "BMY", "ISRG", "VRTX"],
    "US FINANCE": ["JPM", "BAC", "GS", "V", "MA", "WFC", "AMEX",
                   "MS", "CITI", "BLK", "SCHW", "HOOD", "SPGI", "HDFC"],
    "US CONSUMER": ["WMT", "KO", "PEP", "MCD", "NKE", "DIS", "NFLX", "SBUX", "COST", "HD", "PG", "COLG", "ESTEE", "FORD", "GM", "FERRARI", "STELLANTIS", "TOYOTA", "SPOTIFY",
                    "ABNB", "BKNG", "LULU", "CMG", "TGT", "LOW", "MO", "MDLZ"],
    "US INDUSTRIAL": ["BA", "CAT", "XOM", "CVX", "COP",
                      "GE", "HON", "LMT", "RTX", "UNP", "DE", "SLB", "OXY",
                      "VRT", "CEG", "VST", "TLN", "NEE", "OKLO", "GEV"],
    "US SEMI": ["INTC", "QCOM", "AVGO", "MU",
                "MRVL", "SMCI", "MPWR", "TXN", "ADI", "LRCX", "AMAT", "KLAC",
                "NXPI", "ONSEMI", "GFS"],
    "US SOFTWARE": ["CRM", "ORCL", "ADBE", "UBER", "PYPL", "IBM", "CISCO",
                    "NOW", "SNOW", "DDOG", "MDB", "NET", "PANW", "CRWD", "ZS",
                    "TEAM", "SHOP", "INTU", "WDAY"],
    "EU INDICES": ["DAX", "CAC40", "ESTOXX50", "FTSE100", "IBEX35", "FTSEMIB", "AEX", "SMI"],
    "EU STOCKS": ["ASML", "LVMH", "SAP", "NESTLE", "NOVO", "AZN", "SHELL", "TOTAL",
                  "SIEMENS", "AIRBUS", "LOREAL", "ALLIANZ", "HERMES", "SCHNEIDER",
                  "SANTANDER", "BNP", "ENEL", "IBERDROLA",
                  "VW", "MERCEDES", "BMW", "PORSCHE", "ADIDAS", "KERING", "RICHEMONT", "UNILEVER", "DIAGEO", "ABINBEV"],
    "RUS BLUE CHIPS": ["SBER", "GAZP", "LKOH", "ROSN", "NVTK", "TATN", "SNGS", "PLZL", "SIBN", "MGNT"],
    "RUS FINANCE": ["TCSG", "VTBR", "BSPB", "MOEX_EX", "CBOM"],
    "RUS TECH": ["YNDX", "OZON", "VKCO", "POSI", "MTSS", "RTKM", "HHRU", "SOFL", "ASTR", "WUSH"],
    "RUS METALS": ["CHMF", "NLMK", "MAGN", "RUAL", "ALRS", "TRMK", "MTLR", "RASP"],
    "RUS INFRA": ["IRAO", "HYDR", "FLOT", "AFLT", "PIKK", "FEES", "UPRO", "MSNG", "NMTP"],
    "RUS CONSUMER": ["FIVE", "FIXP", "LENT", "MVID"],
    "RUS PROPERTY": ["SMLT", "LSRG", "PHOR", "SGZH"],
    "FOREX MAJORS": ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD", "USDRUB"],
    "FOREX CROSSES": ["EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURCAD", "EURNZD",
                      "GBPJPY", "GBPAUD", "GBPCAD", "GBPCHF", "GBPNZD",
                      "AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD",
                      "CADJPY", "CHFJPY", "NZDJPY"],
    "FOREX EXOTIC": ["USDTRY", "USDMXN", "USDZAR", "USDSGD", "USDNOK", "USDSEK", "USDPLN", "USDCNH"],
}


# --- 3b. CONSOLE RADAR GROUPS (predict.py) ---
# Coarser groups for console output. Assembled from ASSET_TYPES so that adding an
# asset does not require a second list that you have to remember to update.

def _merge_types(*keys: str) -> list:
    out = []
    for k in keys:
        out.extend(ASSET_TYPES[k])
    return out


RADAR_GROUPS = {
    "INDICES & MACRO": [a for a in ASSET_TYPES["INDICES & MACRO"] if a != "IMOEX"],
    "COMMODITIES":     ASSET_TYPES["COMMODITIES"],
    "US TECH":         ASSET_TYPES["US TECH"],
    "US HEALTHCARE":   ASSET_TYPES["US HEALTHCARE"],
    "US FINANCE":      ASSET_TYPES["US FINANCE"],
    "US CONSUMER":     ASSET_TYPES["US CONSUMER"],
    "US INDUSTRIAL":   ASSET_TYPES["US INDUSTRIAL"],
    "US SEMI":         ASSET_TYPES["US SEMI"],
    "US SOFTWARE":     ASSET_TYPES["US SOFTWARE"],
    "EUROPE":          _merge_types("EU INDICES", "EU STOCKS"),
    "CRYPTO":          ASSET_TYPES["CRYPTO"],
    # show IMOEX together with the Russian names
    "MOEX":            ["IMOEX"] + _merge_types(
        "RUS BLUE CHIPS", "RUS FINANCE", "RUS TECH", "RUS METALS",
        "RUS INFRA", "RUS CONSUMER", "RUS PROPERTY"),
    "FOREX":           _merge_types("FOREX MAJORS", "FOREX CROSSES", "FOREX EXOTIC"),
}

# RADAR_GROUPS key maps to a webapp.py chip-cat-* CSS suffix (style.css). Anything not
# listed here (US sectors, indices, forex) falls back to "us" in radar_category().
_CATEGORY_CSS = {"CRYPTO": "crypto", "MOEX": "ru", "COMMODITIES": "commodity", "EUROPE": "eu"}


def radar_category(asset: str) -> str:
    """Coarse category for the webapp's chip-cat accent color: crypto/ru/commodity/us."""
    for group, members in RADAR_GROUPS.items():
        if asset in members:
            return _CATEGORY_CSS.get(group, "us")
    return "us"


# --- 3c. CANONICAL SECTOR MAP ---
# Single source of coarse asset-to-sector membership, DERIVED from ASSET_TYPES so
# adding an asset to its ASSET_TYPES category flows everywhere automatically.
# Consumed by portfolio.py (exposure limits) and sector_rotation.py (rotation),
# replacing the per-module copies they used to keep. Names are display-friendly.
SECTOR_MAP = {
    "Crypto":        ASSET_TYPES["CRYPTO"],
    "US Tech":       ASSET_TYPES["US TECH"],
    "US Health":     ASSET_TYPES["US HEALTHCARE"],
    "US Finance":    ASSET_TYPES["US FINANCE"],
    "US Consumer":   ASSET_TYPES["US CONSUMER"],
    "US Industrial": _merge_types("US INDUSTRIAL", "US SEMI", "US SOFTWARE"),
    "Indices":       [a for a in ASSET_TYPES["INDICES & MACRO"]
                      if a not in ("VIX", "DXY", "TNX", "IMOEX")] + ASSET_TYPES["EU INDICES"],
    "Macro":         ["VIX", "DXY", "TNX"],
    "Commodities":   ASSET_TYPES["COMMODITIES"],
    "Europe":        ASSET_TYPES["EU STOCKS"],
    "Russia":        ["IMOEX"] + _merge_types(
        "RUS BLUE CHIPS", "RUS FINANCE", "RUS TECH", "RUS METALS",
        "RUS INFRA", "RUS CONSUMER", "RUS PROPERTY"),
    "Forex":         _merge_types("FOREX MAJORS", "FOREX CROSSES", "FOREX EXOTIC"),
}


# --- 4. ENVIRONMENT VALIDATION ---
# Importing config.py must not fail when secrets are unset: most scripts
# (data_engine, train_hybrid, predict, backtest) work without Telegram. Scripts
# that really need credentials call require_env(...) at startup and get a clear
# error instead of failing somewhere deep in network code.

class ConfigError(RuntimeError):
    """A required environment variable is not set."""


def require_env(*names: str) -> None:
    """Fail with a clear message if any of the variables is empty.

    Example: require_env("TELEGRAM_TOKEN", "TELEGRAM_USER_ID") at the start of alert_bot.py.
    """
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        raise ConfigError(
            "Environment variables not set: " + ", ".join(missing) + ". "
            "Copy .env.example to .env and fill in the values."
        )


def validate_telegram_config() -> None:
    """Check that Telegram credentials are set. Call before sending alerts."""
    require_env("TELEGRAM_TOKEN", "TELEGRAM_USER_ID")
