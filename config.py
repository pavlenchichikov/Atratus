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
    'AAVE': 'AAVE-USD', 'OP': 'OP-USD',
    # Yahoo disambiguates these with a numeric suffix, the bare symbol returns nothing
    # POL joined them at the MATIC rename: bare POL-USD is a different token that
    # stopped in 2023, and MATIC-USD was retired after 2025-03-24. POL28321-USD
    # starts 2023-10-30, one day inside the history already stored, so it appends
    # without a gap.
    'APT': 'APT21794-USD', 'ARB': 'ARB11841-USD', 'SUI': 'SUI20947-USD',
    'POL': 'POL28321-USD',
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

    # US TREASURY YIELDS - the rest of the curve beside TNX
    'FVX': '^FVX', 'IRX': '^IRX', 'TYX': '^TYX',

    # VOLATILITY INDICES - equity, oil and gold implied vol
    'GVZ': '^GVZ', 'OVX': '^OVX', 'VVIX': '^VVIX', 'VXD': '^VXD',
    'VXN': '^VXN',

    # WORLD INDICES (extra)
    'ATX': '^ATX', 'BEL20': '^BFX', 'DJT': '^DJT', 'DJU': '^DJU',
    'EGX30': '^CASE30', 'EURONEXT100': '^N100', 'JAKARTA': '^JKSE',
    'KLCI': '^KLSE', 'MERVAL': '^MERV', 'MEXBOL': '^MXX',
    'MSCIWORLD': '^990100-USD-STRD', 'NASDAQ100': '^NDX', 'NYSE': '^NYA',
    'NZ50': '^NZ50', 'OMXS30': '^OMX', 'PSI20': 'PSI20.LS', 'SENSEX': '^BSESN',
    'SP400': '^MID', 'STI': '^STI', 'STOXX600': '^STOXX', 'TA125': '^TA125.TA',
    'TAIEX': '^TWII', 'WIG20': 'WIG20.WA', 'WILSHIRE': '^W5000',

    # BONDS AND CREDIT, through the liquid ETFs
    'AGG': 'AGG', 'BND': 'BND', 'BNDX': 'BNDX', 'EMB': 'EMB', 'HYG': 'HYG',
    'IEF': 'IEF', 'JNK': 'JNK', 'LQD': 'LQD', 'MUB': 'MUB', 'SHV': 'SHV',
    'SHY': 'SHY', 'TIP': 'TIP', 'TLH': 'TLH', 'TLT': 'TLT', 'VCIT': 'VCIT',

    # US SECTORS - the eleven SPDRs
    'XLB': 'XLB', 'XLC': 'XLC', 'XLE': 'XLE', 'XLF': 'XLF', 'XLI': 'XLI',
    'XLK': 'XLK', 'XLP': 'XLP', 'XLRE': 'XLRE', 'XLU': 'XLU', 'XLV': 'XLV',
    'XLY': 'XLY',

    # BROAD AND COUNTRY ETFS
    'DIA': 'DIA', 'EEM': 'EEM', 'EFA': 'EFA', 'EWA': 'EWA', 'EWC': 'EWC',
    'EWG': 'EWG', 'EWH': 'EWH', 'EWJ': 'EWJ', 'EWT': 'EWT', 'EWU': 'EWU',
    'EWW': 'EWW', 'EWY': 'EWY', 'EWZ': 'EWZ', 'EZA': 'EZA', 'FXI': 'FXI',
    'INDA': 'INDA', 'IWM': 'IWM', 'KWEB': 'KWEB', 'QQQ': 'QQQ', 'SPY': 'SPY',
    'TUR': 'TUR', 'VEA': 'VEA', 'VTI': 'VTI', 'VWO': 'VWO',

    # THEMATIC ETFS
    'ARKK': 'ARKK', 'GDX': 'GDX', 'GDXJ': 'GDXJ', 'IBB': 'IBB', 'IBIT': 'IBIT',
    'ICLN': 'ICLN', 'ITB': 'ITB', 'IYR': 'IYR', 'JETS': 'JETS', 'LIT': 'LIT',
    'MOAT': 'MOAT', 'PAVE': 'PAVE', 'SCHD': 'SCHD', 'SMH': 'SMH',
    'SOXX': 'SOXX', 'TAN': 'TAN', 'URA': 'URA', 'VNQ': 'VNQ', 'XAR': 'XAR',
    'XBI': 'XBI',

    # COMMODITY AND CURRENCY ETFS
    'CORN_ETF': 'CORN', 'CPER': 'CPER', 'DBA': 'DBA', 'DBC': 'DBC',
    'FXE': 'FXE', 'FXY': 'FXY', 'GLD': 'GLD', 'PDBC': 'PDBC', 'PPLT': 'PPLT',
    'SLV': 'SLV', 'UNG': 'UNG', 'USO': 'USO', 'UUP': 'UUP', 'WEAT': 'WEAT',

    # COMMODITIES (extra)
    'ALUMINIUM': 'ALI=F', 'CATTLE': 'LE=F', 'COTTON': 'CT=F', 'FEEDER': 'GF=F',
    'GASOLINE': 'RB=F', 'HEATOIL': 'HO=F', 'HOGS': 'HE=F', 'KCWHEAT': 'KE=F',
    'LUMBER': 'LBR=F', 'MICROGOLD': 'MGC=F', 'OATS': 'ZO=F',
    'ORANGEJUICE': 'OJ=F', 'RICE': 'ZR=F', 'SOYMEAL': 'ZM=F', 'SOYOIL': 'ZL=F',

    # US STOCKS (extra)
    'AA': 'AA', 'AAL': 'AAL', 'ADM': 'ADM', 'AEE': 'AEE', 'AEP': 'AEP',
    # ALLSTATE, not ALL: the table name is the key lowercased, `all` is a SQL
    # keyword, and this project interpolates table names unquoted in dozens of
    # queries. A test refuses any key that cannot be a table name.
    'AFL': 'AFL', 'AFRM': 'AFRM', 'AIG': 'AIG', 'ALLSTATE': 'ALL',
    'ALLY': 'ALLY',
    'AMT': 'AMT', 'AON': 'AON', 'APD': 'APD', 'APH': 'APH', 'VMRK': 'VMRK',
    'AZO': 'AZO', 'BE': 'BE', 'BEN': 'BEN', 'BG': 'BG', 'BIIB': 'BIIB',
    'BRK-B': 'BRK-B', 'BSX': 'BSX', 'CB': 'CB', 'CCI': 'CCI', 'CCL': 'CCL',
    'CF': 'CF', 'CFG': 'CFG', 'CHPT': 'CHPT', 'CI': 'CI', 'CLF': 'CLF',
    'CME': 'CME', 'CMI': 'CMI', 'CMS': 'CMS', 'COF': 'COF', 'CSX': 'CSX',
    'CVS': 'CVS', 'D': 'D', 'DAL': 'DAL', 'DASH': 'DASH', 'DD': 'DD',
    'DG': 'DG', 'DHR': 'DHR', 'DLR': 'DLR', 'DLTR': 'DLTR', 'DOWINC': 'DOW',
    'DPZ': 'DPZ', 'DTE': 'DTE', 'DUK': 'DUK', 'ECL': 'ECL', 'ED': 'ED',
    'ELV': 'ELV', 'EMR': 'EMR', 'ENPH': 'ENPH', 'EQIX': 'EQIX',
    'ES': 'ES', 'ETN': 'ETN', 'ETR': 'ETR', 'EVRG': 'EVRG', 'EXC': 'EXC',
    'FAST': 'FAST', 'FCX': 'FCX', 'FDX': 'FDX', 'FE': 'FE', 'FITB': 'FITB',
    'FSLR': 'FSLR', 'GD': 'GD', 'GIS': 'GIS', 'GLW': 'GLW', 'GWW': 'GWW',
    'HBAN': 'HBAN', 'HLT': 'HLT', 'HOG': 'HOG', 'HSY': 'HSY', 'HUM': 'HUM',
    'HWM': 'HWM', 'ICE': 'ICE', 'ITW': 'ITW', 'IVZ': 'IVZ', 'KEY': 'KEY',
    'KEYS': 'KEYS', 'KHC': 'KHC', 'KMB': 'KMB', 'KR': 'KR', 'LCID': 'LCID',
    'LHX': 'LHX', 'LIN': 'LIN', 'LNT': 'LNT', 'LUV': 'LUV', 'LYFT': 'LYFT',
    'LYV': 'LYV', 'MAR': 'MAR', 'MCK': 'MCK', 'MCO': 'MCO', 'MDT': 'MDT',
    'MET': 'MET', 'MLM': 'MLM', 'MMM': 'MMM', 'MOS': 'MOS', 'MRNA': 'MRNA',
    'MSCI': 'MSCI', 'NCLH': 'NCLH', 'NDAQ': 'NDAQ', 'NEM': 'NEM', 'NI': 'NI',
    'NOC': 'NOC', 'NSC': 'NSC', 'NTRS': 'NTRS', 'NUE': 'NUE', 'O': 'O',
    'ORLY': 'ORLY', 'PARA': 'PARA', 'PCAR': 'PCAR', 'PCG': 'PCG', 'PEG': 'PEG',
    'PGR': 'PGR', 'PH': 'PH', 'PINS': 'PINS', 'PLD': 'PLD', 'PLUG': 'PLUG',
    'PNC': 'PNC', 'PPG': 'PPG', 'PPL': 'PPL', 'PRU': 'PRU', 'PSA': 'PSA',
    'RBLX': 'RBLX', 'RCL': 'RCL', 'REGN': 'REGN', 'RF': 'RF', 'RIVN': 'RIVN',
    # ROSS is Ross Stores on Yahoo; the MOEX ROST further down is a different
    # company. Routing is by KEY through MOEX_ASSETS, so the shared symbol
    # string is not a collision.
    'ROK': 'ROK', 'ROSS': 'ROST', 'RSG': 'RSG', 'SBAC': 'SBAC',
    'SEDG': 'SEDG',
    'SHW': 'SHW', 'SNAP': 'SNAP', 'SO': 'SO', 'SOFI': 'SOFI', 'SPG': 'SPG',
    'SRE': 'SRE', 'STT': 'STT', 'SYK': 'SYK', 'SYY': 'SYY', 'TDG': 'TDG',
    'TEL': 'TEL', 'TER': 'TER', 'TFC': 'TFC', 'TJX': 'TJX', 'TROW': 'TROW',
    'TRV': 'TRV', 'TSN': 'TSN', 'TTWO': 'TTWO', 'U': 'U', 'UAL': 'UAL',
    'UPS': 'UPS', 'URI': 'URI', 'USB': 'USB', 'VMC': 'VMC', 'VTR': 'VTR',
    'WBD': 'WBD', 'WEC': 'WEC', 'WELL': 'WELL', 'WM': 'WM',
    'XEL': 'XEL', 'YUM': 'YUM', 'ZION': 'ZION', 'ZTS': 'ZTS',

    # EU STOCKS (extra) - key is the native listing on Yahoo
    'ABB': 'ABBN.SW', 'ADYEN': 'ADYEN.AS', 'AENA': 'AENA.MC', 'AHOLD': 'AD.AS',
    'AIRLIQUIDE': 'AI.PA', 'ANGLO': 'AAL.L', 'ASSAABLOY': 'ASSA-B.ST',
    'ATLASCOPCO': 'ATCO-A.ST', 'BARCLAYS': 'BARC.L', 'BASF': 'BAS.DE',
    'BAT': 'BATS.L', 'BAYER': 'BAYN.DE', 'BBVA': 'BBVA.MC', 'BP': 'BP.L',
    'CAPGEMINI': 'CAP.PA', 'CARREFOUR': 'CA.PA', 'CONTI': 'CON.DE',
    'CREDITAG': 'ACA.PA', 'DANONE': 'BN.PA', 'DEUTSCHEBANK': 'DBK.DE',
    'DHL': 'DHL.DE', 'DNB': 'DNB.OL', 'DSM': 'DSFIR.AS', 'DSV': 'DSV.CO',
    'DTELEKOM': 'DTE.DE', 'ENGIE': 'ENGI.PA', 'ENI': 'ENI.MI',
    'EON': 'EOAN.DE', 'EQUINOR': 'EQNR.OL', 'ERICSSON': 'ERIC-B.ST',
    'ERSTE': 'EBS.VI', 'ESSILOR': 'EL.PA', 'FORTUM': 'FORTUM.HE',
    'GENERALI': 'G.MI', 'GIVAUDAN': 'GIVN.SW', 'GLENCORE': 'GLEN.L',
    'GSK': 'GSK.L', 'HEINEKEN': 'HEIA.AS', 'HM': 'HM-B.ST', 'HSBC': 'HSBA.L',
    'INDITEX': 'ITX.MC', 'INFINEON': 'IFX.DE', 'ING': 'INGA.AS',
    'INTESA': 'ISP.MI', 'INVESTOR': 'INVE-B.ST', 'KBC': 'KBC.BR',
    'KPN': 'KPN.AS', 'LLOYDS': 'LLOY.L', 'LSEG': 'LSEG.L',
    'MAERSK': 'MAERSK-B.CO', 'MERCKDE': 'MRK.DE', 'MICHELIN': 'ML.PA',
    'MUNICHRE': 'MUV2.DE', 'NATGRID': 'NG.L', 'NOKIA': 'NOKIA.HE',
    'NORDEA': 'NDA-FI.HE', 'NORSKHYDRO': 'NHY.OL', 'NOVARTIS': 'NOVN.SW',
    'OMV': 'OMV.VI', 'ORANGE': 'ORA.PA', 'ORSTED': 'ORSTED.CO',
    'PERNOD': 'RI.PA', 'PHILIPS': 'PHIA.AS', 'PROSUS': 'PRX.AS',
    'RANDSTAD': 'RAND.AS', 'RELX': 'REL.L', 'REPSOL': 'REP.MC', 'RIO': 'RIO.L',
    'ROLLSROYCE': 'RR.L', 'RWE': 'RWE.DE', 'SAINTGOBAIN': 'SGO.PA',
    'SAMPO': 'SAMPO.HE', 'SANDVIK': 'SAND.ST', 'SANOFI': 'SAN.PA',
    'SEB': 'SEB-A.ST', 'SIKA': 'SIKA.SW', 'SOCGEN': 'GLE.PA',
    'SOLVAY': 'SOLB.BR', 'SWISSRE': 'SREN.SW', 'TELECOMIT': 'TIT.MI',
    'TELEFONICA': 'TEF.MC', 'TELENOR': 'TEL.OL', 'TESCO': 'TSCO.L',
    'UBS': 'UBSG.SW', 'UCB': 'UCB.BR', 'UNICREDIT': 'UCG.MI', 'UPM': 'UPM.HE',
    'VEOLIA': 'VIE.PA', 'VESTAS': 'VWS.CO', 'VINCI': 'DG.PA',
    'VODAFONE': 'VOD.L', 'VOLVO': 'VOLV-B.ST', 'WOLTERS': 'WKL.AS',
    'ZALANDO': 'ZAL.DE', 'ZURICH': 'ZURN.SW',

    # RU MARKET (extra) - preferred shares, mid and small caps
    'ABIO': 'ABIO', 'ABRD': 'ABRD', 'AFKS': 'AFKS', 'AKRN': 'AKRN',
    'APTK': 'APTK', 'AQUA': 'AQUA', 'ARSA': 'ARSA', 'ASSB': 'ASSB',
    'AVAN': 'AVAN', 'BANE': 'BANE', 'BANEP': 'BANEP', 'BELU': 'BELU',
    'BRZL': 'BRZL', 'BSPBP': 'BSPBP', 'CARM': 'CARM', 'CHMK': 'CHMK',
    'CNTL': 'CNTL', 'CNTLP': 'CNTLP', 'DIAS': 'DIAS', 'DVEC': 'DVEC',
    'DZRD': 'DZRD', 'ELFV': 'ELFV', 'ENPG': 'ENPG', 'ETLN': 'ETLN',
    'EUTR': 'EUTR', 'GAZA': 'GAZA', 'GAZAP': 'GAZAP', 'GCHE': 'GCHE',
    'GEMA': 'GEMA', 'GEMC': 'GEMC', 'GTRK': 'GTRK', 'HNFG': 'HNFG',
    'IGST': 'IGST', 'IRKT': 'IRKT', 'IVAT': 'IVAT', 'KAZT': 'KAZT',
    'KAZTP': 'KAZTP', 'KCHE': 'KCHE', 'KGKC': 'KGKC', 'KLVZ': 'KLVZ',
    'KMAZ': 'KMAZ', 'KRKNP': 'KRKNP', 'KROT': 'KROT', 'KRSB': 'KRSB',
    'KUZB': 'KUZB', 'KZOS': 'KZOS', 'KZOSP': 'KZOSP', 'LEAS': 'LEAS',
    'LIFE': 'LIFE', 'LSNG': 'LSNG', 'LSNGP': 'LSNGP', 'MAGE': 'MAGE',
    'MBNK': 'MBNK', 'MDMG': 'MDMG', 'MFGS': 'MFGS', 'MGKL': 'MGKL',
    'MGTS': 'MGTS', 'MGTSP': 'MGTSP', 'MISB': 'MISB', 'MRKC': 'MRKC',
    'MRKK': 'MRKK', 'MRKP': 'MRKP', 'MRKS': 'MRKS', 'MRKU': 'MRKU',
    'MRKV': 'MRKV', 'MRKY': 'MRKY', 'MRKZ': 'MRKZ', 'MSRS': 'MSRS',
    'MTLRP': 'MTLRP', 'NAUK': 'NAUK', 'NFAZ': 'NFAZ', 'NKHP': 'NKHP',
    'NKNC': 'NKNC', 'NKNCP': 'NKNCP', 'NSVZ': 'NSVZ', 'OGKB': 'OGKB',
    'OKEY': 'OKEY', 'PAZA': 'PAZA', 'PMSB': 'PMSB', 'PRFN': 'PRFN',
    'PRMD': 'PRMD', 'RBCM': 'RBCM', 'RENI': 'RENI', 'RKKE': 'RKKE',
    'RNFT': 'RNFT', 'ROLO': 'ROLO', 'ROST': 'ROST', 'RTGZ': 'RTGZ',
    'RTKMP': 'RTKMP', 'RTSB': 'RTSB', 'RUSI': 'RUSI', 'RZSB': 'RZSB',
    'SARE': 'SARE', 'SBERP': 'SBERP', 'SELG': 'SELG', 'SFIN': 'SFIN',
    'SLEN': 'SLEN', 'SNGSP': 'SNGSP', 'SPBE': 'SPBE', 'STSB': 'STSB',
    'SVAV': 'SVAV', 'SVCB': 'SVCB', 'SVET': 'SVET', 'TASB': 'TASB',
    'TATNP': 'TATNP', 'TGKA': 'TGKA', 'TGKB': 'TGKB', 'TGKN': 'TGKN',
    'TNSE': 'TNSE', 'TORS': 'TORS', 'TRNFP': 'TRNFP', 'TTLK': 'TTLK',
    'UGLD': 'UGLD', 'UKUZ': 'UKUZ', 'UNAC': 'UNAC', 'URKZ': 'URKZ',
    'USBN': 'USBN', 'UWGN': 'UWGN', 'VGSB': 'VGSB', 'VJGZ': 'VJGZ',
    'VLHZ': 'VLHZ', 'VRSB': 'VRSB', 'VSEH': 'VSEH', 'VSMO': 'VSMO',
    'YAKG': 'YAKG', 'YKEN': 'YKEN', 'YRSB': 'YRSB', 'ZAYM': 'ZAYM',
    'ZILL': 'ZILL', 'ZVEZ': 'ZVEZ',
}

# Keys fetched from MOEX rather than Yahoo. data_engine reads this; it used to
# keep a second copy of the list, and a ticker added to FULL_ASSET_MAP but not
# to that copy was fetched from Yahoo, returned nothing and stayed stale.
MOEX_ASSETS = [
    "IMOEX", "SBER", "GAZP", "LKOH", "ROSN", "NVTK", "TATN", "SNGS", "PLZL",
    "SIBN", "MGNT", "TCSG", "VTBR", "BSPB", "MOEX_EX", "CBOM", "YNDX", "OZON",
    "VKCO", "POSI", "MTSS", "RTKM", "HHRU", "SOFL", "ASTR", "WUSH", "CHMF",
    "NLMK", "MAGN", "RUAL", "ALRS", "TRMK", "MTLR", "RASP", "IRAO", "HYDR",
    "FLOT", "AFLT", "PIKK", "FEES", "UPRO", "MSNG", "NMTP", "PHOR", "SGZH",
    "FIVE", "FIXP", "LENT", "MVID", "SMLT", "LSRG", "ABIO", "ABRD", "AFKS",
    "AKRN", "APTK", "AQUA", "ARSA", "ASSB", "AVAN", "BANE", "BANEP", "BELU",
    "BRZL", "BSPBP", "CARM", "CHMK", "CNTL", "CNTLP", "DIAS", "DVEC", "DZRD",
    "ELFV", "ENPG", "ETLN", "EUTR", "GAZA", "GAZAP", "GCHE", "GEMA", "GEMC",
    "GTRK", "HNFG", "IGST", "IRKT", "IVAT", "KAZT", "KAZTP", "KCHE", "KGKC",
    "KLVZ", "KMAZ", "KRKNP", "KROT", "KRSB", "KUZB", "KZOS", "KZOSP", "LEAS",
    "LIFE", "LSNG", "LSNGP", "MAGE", "MBNK", "MDMG", "MFGS", "MGKL", "MGTS",
    "MGTSP", "MISB", "MRKC", "MRKK", "MRKP", "MRKS", "MRKU", "MRKV", "MRKY",
    "MRKZ", "MSRS", "MTLRP", "NAUK", "NFAZ", "NKHP", "NKNC", "NKNCP", "NSVZ",
    "OGKB", "OKEY", "PAZA", "PMSB", "PRFN", "PRMD", "RBCM", "RENI", "RKKE",
    "RNFT", "ROLO", "ROST", "RTGZ", "RTKMP", "RTSB", "RUSI", "RZSB", "SARE",
    "SBERP", "SELG", "SFIN", "SLEN", "SNGSP", "SPBE", "STSB", "SVAV", "SVCB",
    "SVET", "TASB", "TATNP", "TGKA", "TGKB", "TGKN", "TNSE", "TORS", "TRNFP",
    "TTLK", "UGLD", "UKUZ", "UNAC", "URKZ", "USBN", "UWGN", "VGSB", "VJGZ",
    "VLHZ", "VRSB", "VSEH", "VSMO", "YAKG", "YKEN", "YRSB", "ZAYM", "ZILL",
    "ZVEZ",
]

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
    # Classes for the 2026-08-25 additions. Every asset needs one: the holdout
    # is grown balanced across these, and an asset outside every class falls
    # into a single unnamed bucket that then dominates the draw.
    "RATES": [
        "FVX", "IRX", "TYX"],
    "VOLATILITY": [
        "GVZ", "OVX", "VVIX", "VXD", "VXN"],
    "WORLD INDICES": [
        "ATX", "BEL20", "DJT", "DJU", "EGX30", "EURONEXT100", "JAKARTA",
        "KLCI", "MERVAL", "MEXBOL", "MSCIWORLD", "NASDAQ100", "NYSE", "NZ50",
        "OMXS30", "PSI20", "SENSEX", "SP400", "STI", "STOXX600", "TA125",
        "TAIEX", "WIG20", "WILSHIRE"],
    "BOND ETFS": [
        "AGG", "BND", "BNDX", "EMB", "HYG", "IEF", "JNK", "LQD", "MUB",
        "SHV", "SHY", "TIP", "TLH", "TLT", "VCIT"],
    "SECTOR ETFS": [
        "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU",
        "XLV", "XLY"],
    "BROAD ETFS": [
        "DIA", "EEM", "EFA", "EWA", "EWC", "EWG", "EWH", "EWJ", "EWT", "EWU",
        "EWW", "EWY", "EWZ", "EZA", "FXI", "INDA", "IWM", "KWEB", "QQQ",
        "SPY", "TUR", "VEA", "VTI", "VWO"],
    "THEME ETFS": [
        "ARKK", "GDX", "GDXJ", "IBB", "IBIT", "ICLN", "ITB", "IYR", "JETS",
        "LIT", "MOAT", "PAVE", "SCHD", "SMH", "SOXX", "TAN", "URA", "VNQ",
        "XAR", "XBI"],
    "COMMODITY ETFS": [
        "CORN_ETF", "CPER", "DBA", "DBC", "FXE", "FXY", "GLD", "PDBC",
        "PPLT", "SLV", "UNG", "USO", "UUP", "WEAT"],
    "COMMODITIES EXTRA": [
        "ALUMINIUM", "CATTLE", "COTTON", "FEEDER", "GASOLINE", "HEATOIL",
        "HOGS", "KCWHEAT", "LUMBER", "MICROGOLD", "OATS", "ORANGEJUICE",
        "RICE", "SOYMEAL", "SOYOIL"],
    "US EXTRA": [
        "AA", "AAL", "ADM", "AEE", "AEP", "AFL", "AFRM", "AIG", "ALLSTATE",
        "ALLY", "AMT", "AON", "APD", "APH", "AZO", "BE", "BEN", "BG",
        "BIIB", "BRK-B", "BSX", "CB", "CCI", "CCL", "CF", "CFG", "CHPT",
        "CI", "CLF", "CME", "CMI", "CMS", "COF", "CSX", "CVS", "D", "DAL",
        "DASH", "DD", "DG", "DHR", "DLR", "DLTR", "DOWINC", "DPZ", "DTE",
        "DUK", "ECL", "ED", "ELV", "EMR", "ENPH", "EQIX", "ES", "ETN",
        "ETR", "EVRG", "EXC", "FAST", "FCX", "FDX", "FE", "FITB", "FSLR",
        "GD", "GIS", "GLW", "GWW", "HBAN", "HLT", "HOG", "HSY", "HUM", "HWM",
        "ICE", "ITW", "IVZ", "KEY", "KEYS", "KHC", "KMB", "KR", "LCID",
        "LHX", "LIN", "LNT", "LUV", "LYFT", "LYV", "MAR", "MCK", "MCO",
        "MDT", "MET", "MLM", "MMM", "MOS", "MRNA", "MSCI", "NCLH", "NDAQ",
        "NEM", "NI", "NOC", "NSC", "NTRS", "NUE", "O", "ORLY", "PARA",
        "PCAR", "PCG", "PEG", "PGR", "PH", "PINS", "PLD", "PLUG", "PNC",
        "PPG", "PPL", "PRU", "PSA", "RBLX", "RCL", "REGN", "RF", "RIVN",
        "ROK", "ROSS", "RSG", "SBAC", "SEDG", "SHW", "SNAP", "SO", "SOFI",
        "SPG", "SRE", "STT", "SYK", "SYY", "TDG", "TEL", "TER", "TFC", "TJX",
        "TROW", "TRV", "TSN", "TTWO", "U", "UAL", "UPS", "URI", "USB", "VMC",
        "VMRK", "VTR", "WBD", "WEC", "WELL", "WM", "XEL", "YUM", "ZION",
        "ZTS"],
    "EU EXTRA": [
        "ABB", "ADYEN", "AENA", "AHOLD", "AIRLIQUIDE", "ANGLO", "ASSAABLOY",
        "ATLASCOPCO", "BARCLAYS", "BASF", "BAT", "BAYER", "BBVA", "BP",
        "CAPGEMINI", "CARREFOUR", "CONTI", "CREDITAG", "DANONE",
        "DEUTSCHEBANK", "DHL", "DNB", "DSM", "DSV", "DTELEKOM", "ENGIE",
        "ENI", "EON", "EQUINOR", "ERICSSON", "ERSTE", "ESSILOR", "FORTUM",
        "GENERALI", "GIVAUDAN", "GLENCORE", "GSK", "HEINEKEN", "HM", "HSBC",
        "INDITEX", "INFINEON", "ING", "INTESA", "INVESTOR", "KBC", "KPN",
        "LLOYDS", "LSEG", "MAERSK", "MERCKDE", "MICHELIN", "MUNICHRE",
        "NATGRID", "NOKIA", "NORDEA", "NORSKHYDRO", "NOVARTIS", "OMV",
        "ORANGE", "ORSTED", "PERNOD", "PHILIPS", "PROSUS", "RANDSTAD",
        "RELX", "REPSOL", "RIO", "ROLLSROYCE", "RWE", "SAINTGOBAIN", "SAMPO",
        "SANDVIK", "SANOFI", "SEB", "SIKA", "SOCGEN", "SOLVAY", "SWISSRE",
        "TELECOMIT", "TELEFONICA", "TELENOR", "TESCO", "UBS", "UCB",
        "UNICREDIT", "UPM", "VEOLIA", "VESTAS", "VINCI", "VODAFONE", "VOLVO",
        "WOLTERS", "ZALANDO", "ZURICH"],
    "RUS EXTRA": [
        "ABIO", "ABRD", "AFKS", "AKRN", "APTK", "AQUA", "ARSA", "ASSB",
        "AVAN", "BANE", "BANEP", "BELU", "BRZL", "BSPBP", "CARM", "CHMK",
        "CNTL", "CNTLP", "DIAS", "DVEC", "DZRD", "ELFV", "ENPG", "ETLN",
        "EUTR", "GAZA", "GAZAP", "GCHE", "GEMA", "GEMC", "GTRK", "HNFG",
        "IGST", "IRKT", "IVAT", "KAZT", "KAZTP", "KCHE", "KGKC", "KLVZ",
        "KMAZ", "KRKNP", "KROT", "KRSB", "KUZB", "KZOS", "KZOSP", "LEAS",
        "LIFE", "LSNG", "LSNGP", "MAGE", "MBNK", "MDMG", "MFGS", "MGKL",
        "MGTS", "MGTSP", "MISB", "MRKC", "MRKK", "MRKP", "MRKS", "MRKU",
        "MRKV", "MRKY", "MRKZ", "MSRS", "MTLRP", "NAUK", "NFAZ", "NKHP",
        "NKNC", "NKNCP", "NSVZ", "OGKB", "OKEY", "PAZA", "PMSB", "PRFN",
        "PRMD", "RBCM", "RENI", "RKKE", "RNFT", "ROLO", "ROST", "RTGZ",
        "RTKMP", "RTSB", "RUSI", "RZSB", "SARE", "SBERP", "SELG", "SFIN",
        "SLEN", "SNGSP", "SPBE", "STSB", "SVAV", "SVCB", "SVET", "TASB",
        "TATNP", "TGKA", "TGKB", "TGKN", "TNSE", "TORS", "TRNFP", "TTLK",
        "UGLD", "UKUZ", "UNAC", "URKZ", "USBN", "UWGN", "VGSB", "VJGZ",
        "VLHZ", "VRSB", "VSEH", "VSMO", "YAKG", "YKEN", "YRSB", "ZAYM",
        "ZILL", "ZVEZ"],
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
    "EUROPE":          _merge_types("EU INDICES", "EU STOCKS", "EU EXTRA"),
    "CRYPTO":          ASSET_TYPES["CRYPTO"],
    # show IMOEX together with the Russian names
    # RUS EXTRA belongs here and was missing. radar_category falls back to "us"
    # for anything unlisted, so 130 second-tier Moscow names read as American:
    # _is_moex went false, can_have_earnings went TRUE, and the earnings scan
    # asked Yahoo about a bare MOEX ticker and bought one 404 each - the failure
    # 844a70b fixed for SBER and VTBR and never reached these. They also lost
    # their Smart-Lab fundamentals and wore the wrong accent on the radar.
    "MOEX":            ["IMOEX"] + _merge_types(
        "RUS BLUE CHIPS", "RUS FINANCE", "RUS TECH", "RUS METALS",
        "RUS INFRA", "RUS CONSUMER", "RUS PROPERTY", "RUS EXTRA"),
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
    # Added 2026-08-31. SECTOR_MAP named 12 of the 35 ASSET_TYPES categories and
    # stopped being extended as the map grew, so 523 of 847 assets belonged to no
    # sector at all: invisible to the rotation heatmap and pooled into portfolio's
    # OTHER bucket for exposure. The EXTRA categories fold into the region they
    # belong to; the rest are their own row because an ETF sleeve is not a sector
    # of the equity market and averaging it into one would say nothing.
    "Rates":         ASSET_TYPES["RATES"],
    "Volatility":    ASSET_TYPES["VOLATILITY"],
    "World Indices": ASSET_TYPES["WORLD INDICES"],
    "Bond ETFs":     ASSET_TYPES["BOND ETFS"],
    "Sector ETFs":   ASSET_TYPES["SECTOR ETFS"],
    "Broad ETFs":    ASSET_TYPES["BROAD ETFS"],
    "Theme ETFs":    ASSET_TYPES["THEME ETFS"],
    "Commodity ETFs": ASSET_TYPES["COMMODITY ETFS"],
    # US EXTRA is 167 names with no recorded sector of their own. One row rather
    # than a guess: splitting them across the existing US sectors would need
    # membership nobody wrote down, and inventing it would make the rotation
    # reading confidently wrong instead of coarse.
    "US Broad":      ASSET_TYPES["US EXTRA"],
}
SECTOR_MAP["Commodities"] = SECTOR_MAP["Commodities"] + ASSET_TYPES["COMMODITIES EXTRA"]
SECTOR_MAP["Europe"] = SECTOR_MAP["Europe"] + ASSET_TYPES["EU EXTRA"]
SECTOR_MAP["Russia"] = SECTOR_MAP["Russia"] + ASSET_TYPES["RUS EXTRA"]


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
