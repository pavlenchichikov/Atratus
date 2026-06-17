"""
DB Audit & Fix - аудит и ремонт market.db

  python db_check.py          - полный read-only аудит (ничего не меняет)
  python db_check.py --fix    - аудит + авто-ремонт исправимых проблем
  python db_check.py --stats  - только статистика таблиц

Аудит делится на два блока:
  ИСПРАВИМОЕ (--fix чинит): дубли по дате, формат даты, скрытые дубли, NULL.
  КАЧЕСТВО ДАННЫХ (read-only, требует внимания/догрузки): свежесть (отставшие
  таблицы), целостность OHLC, разрывы в датах, покрытие по config, мало данных,
  PRAGMA integrity.
"""

import os
import sqlite3
import sys
from datetime import datetime

# Стабильный UTF-8 в консоль/в пайп лаунчера (иначе кириллица бьётся на cp1251).
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "market.db")

# Пороги аудита (можно переопредилить через окружение при желании).
STALE_DAILY_DAYS = 7      # дневная таблица без новых баров дольше - отстала
STALE_WEEKLY_DAYS = 21    # недельная
MAX_GAP_DAYS = 10         # разрыв между соседними дневными барами больше - дыра
GAP_RECENT_DAYS = 730     # дыры старше не показываем (раннюю историю Yahoo даёт разреженно)
MIN_ROWS_TRAIN = 100      # меньше строк - мало для обучения


# -- Утилиты ------------------------------------------------------------------

def _connect():
    if not os.path.exists(DB_PATH):
        print(f"[ERROR] База данных не найдена: {DB_PATH}")
        sys.exit(1)
    return sqlite3.connect(DB_PATH)


def get_tables(cur):
    return sorted(
        r[0]
        for r in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    )


# -- Проверки -----------------------------------------------------------------

def check_duplicates(cur, tables):
    """Таблицы с дубликатами по Date."""
    problems = {}
    for t in tables:
        rows = cur.execute(
            f"SELECT Date, COUNT(*) AS c FROM {t} GROUP BY Date HAVING c > 1"
        ).fetchall()
        if rows:
            extra = sum(r[1] - 1 for r in rows)
            problems[t] = {"dates": len(rows), "extra_rows": extra}
    return problems


def check_date_formats(cur, tables):
    """Даты не в формате YYYY-MM-DD (10 символов)."""
    bad = {}
    for t in tables:
        cnt = cur.execute(
            f"SELECT COUNT(*) FROM {t} WHERE length(Date) != 10"
        ).fetchone()[0]
        if cnt:
            examples = [
                r[0]
                for r in cur.execute(
                    f"SELECT DISTINCT Date FROM {t} WHERE length(Date) != 10 LIMIT 5"
                ).fetchall()
            ]
            bad[t] = {"count": cnt, "examples": examples}
    return bad


def check_hidden_duplicates(cur, tables):
    """Одна дата в разных форматах (скрытые дубли)."""
    hidden = {}
    for t in tables:
        rows = cur.execute(
            f"""SELECT substr(Date,1,10) AS d, COUNT(DISTINCT Date) AS fmts
                FROM {t} GROUP BY d HAVING fmts > 1"""
        ).fetchall()
        if rows:
            hidden[t] = len(rows)
    return hidden


def check_nulls(cur, tables):
    """Строки с NULL в ключевых колонках (Date, Close)."""
    bad = {}
    for t in tables:
        cols = [r[1] for r in cur.execute(f"PRAGMA table_info({t})").fetchall()]
        conditions = []
        if "Date" in cols:
            conditions.append("Date IS NULL")
        for c in cols:
            if c.lower() == "close":
                conditions.append(f"{c} IS NULL")
        if not conditions:
            continue
        cnt = cur.execute(
            f"SELECT COUNT(*) FROM {t} WHERE {' OR '.join(conditions)}"
        ).fetchone()[0]
        if cnt:
            bad[t] = cnt
    return bad


def check_empty_tables(cur, tables):
    """Пустые таблицы (0 строк)."""
    empty = []
    for t in tables:
        cnt = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        if cnt == 0:
            empty.append(t)
    return empty


# -- Аудит качества данных (read-only) ----------------------------------------

def _parse_date(s):
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _price_tables(cur, tables):
    """Только OHLCV-таблицы активов (исключает лог-таблицы guru_log и т.п.)."""
    out = []
    for t in tables:
        cols = {r[1].lower() for r in cur.execute(f"PRAGMA table_info({t})").fetchall()}
        if {"open", "high", "low", "close"} <= cols:
            out.append(t)
    return out


def check_freshness(cur, tables):
    """Сколько дней назад последний бар; отмечает отставшие таблицы.

    Отставшая таблица обычно = делистинг/переименование тикера или сломанный
    фетч (см. историю с FIVE-X5, FIXP-FIXR), а не пустая база."""
    today = datetime.now().date()
    stale = {}
    for t in tables:
        d_max = cur.execute(f"SELECT MAX(Date) FROM {t}").fetchone()[0]
        last = _parse_date(d_max)
        if last is None:
            continue
        age = (today - last).days
        limit = STALE_WEEKLY_DAYS if t.endswith("_weekly") else STALE_DAILY_DAYS
        if age > limit:
            stale[t] = age
    return dict(sorted(stale.items(), key=lambda kv: -kv[1]))


def check_ohlc(cur, tables):
    """Целостность OHLCV, по тяжести.

    critical: High<Low, неположительная цена, отрицательный объём (порча данных).
    minor:    Open/Close вне диапазона [Low,High] - частый артефакт FX-фида Yahoo,
              не порча, но к сведению."""
    out = {}
    for t in tables:
        cols = {r[1].lower() for r in cur.execute(f"PRAGMA table_info({t})").fetchall()}
        if not {"open", "high", "low", "close"} <= cols:
            continue
        crit = ["High < Low", "Open <= 0", "High <= 0", "Low <= 0", "Close <= 0"]
        if "volume" in cols:
            crit.append("Volume < 0")
        minor = ["Close > High", "Close < Low", "Open > High", "Open < Low"]
        c = cur.execute(f"SELECT COUNT(*) FROM {t} WHERE {' OR '.join(crit)}").fetchone()[0]
        m = cur.execute(f"SELECT COUNT(*) FROM {t} WHERE {' OR '.join(minor)}").fetchone()[0]
        if c or m:
            out[t] = {"critical": c, "minor": m}
    return out


def check_gaps(cur, tables):
    """Самый большой разрыв между соседними дневными барами за последние
    GAP_RECENT_DAYS (старую разреженную историю Yahoo не считаем)."""
    from datetime import timedelta
    cutoff = datetime.now().date() - timedelta(days=GAP_RECENT_DAYS)
    gaps = {}
    for t in tables:
        if t.endswith("_weekly"):
            continue
        dates = [r[0] for r in cur.execute(
            f"SELECT DISTINCT substr(Date,1,10) d FROM {t} ORDER BY d"
        ).fetchall()]
        prev = None
        biggest = 0
        span = None
        for ds in dates:
            d = _parse_date(ds)
            if d is None:
                continue
            # считаем дыру только если она заканчивается в недавнем окне
            if prev is not None and d >= cutoff and (d - prev).days > biggest:
                biggest = (d - prev).days
                span = (prev.isoformat(), d.isoformat())
            prev = d
        if biggest > MAX_GAP_DAYS:
            gaps[t] = (biggest, span)
    return dict(sorted(gaps.items(), key=lambda kv: -kv[1][0]))


def check_low_data(cur, tables):
    """Дневные таблицы со слишком малым числом строк для обучения."""
    low = {}
    for t in tables:
        if t.endswith("_weekly"):
            continue
        cnt = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        if 0 < cnt < MIN_ROWS_TRAIN:
            low[t] = cnt
    return dict(sorted(low.items(), key=lambda kv: kv[1]))


def _norm_key(key):
    """Имя таблицы из ключа актива (как в data_engine)."""
    return key.lower().replace("^", "").replace(".", "").replace("-", "")


def check_coverage(tables):
    """Сверка таблиц с реестром активов из config: чего не хватает, что лишнее,
    у каких нет недельной пары. None, если config недоступен."""
    try:
        from config import FULL_ASSET_MAP
    except Exception:
        return None
    expected = {_norm_key(k) for k in FULL_ASSET_MAP}
    present = set(tables)
    daily = {t for t in present if not t.endswith("_weekly")}
    return {
        "missing": sorted(expected - daily),
        "orphan": sorted(daily - expected),
        "no_weekly": sorted(a for a in expected
                            if a in daily and f"{a}_weekly" not in present),
    }


def check_integrity(conn):
    """PRAGMA quick_check + размер файла."""
    res = conn.execute("PRAGMA quick_check").fetchone()[0]
    return {"quick_check": res, "size_mb": os.path.getsize(DB_PATH) / 1024 / 1024}


# -- Исправления --------------------------------------------------------------

def fix_date_formats(cur, tables):
    """Нормализует даты > 10 символов - YYYY-MM-DD."""
    total = 0
    for t in tables:
        before = cur.execute(
            f"SELECT COUNT(*) FROM {t} WHERE length(Date) > 10"
        ).fetchone()[0]
        if before:
            cur.execute(
                f"UPDATE {t} SET Date = substr(Date,1,10) WHERE length(Date) > 10"
            )
            total += before
            print(f"    {t}: нормализовано {before} дат")
    return total


def fix_duplicates(cur, tables):
    """Удаляет дубликаты - оставляет MAX(rowid) для каждой даты."""
    total = 0
    for t in tables:
        before = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        cur.execute(
            f"DELETE FROM {t} WHERE rowid NOT IN "
            f"(SELECT MAX(rowid) FROM {t} GROUP BY Date)"
        )
        after = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        removed = before - after
        if removed:
            total += removed
            print(f"    {t}: удалено {removed} дубликатов")
    return total


def fix_nulls(cur, tables):
    """Удаляет строки с NULL Date или NULL Close."""
    total = 0
    for t in tables:
        cols = [r[1] for r in cur.execute(f"PRAGMA table_info({t})").fetchall()]
        conditions = []
        if "Date" in cols:
            conditions.append("Date IS NULL")
        for c in cols:
            if c.lower() == "close":
                conditions.append(f"{c} IS NULL")
        if not conditions:
            continue
        before = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        cur.execute(f"DELETE FROM {t} WHERE {' OR '.join(conditions)}")
        after = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        removed = before - after
        if removed:
            total += removed
            print(f"    {t}: удалено {removed} строк с NULL")
    return total


def fix_vacuum(conn):
    """VACUUM - сжатие файла БД после удалений."""
    before = os.path.getsize(DB_PATH)
    conn.execute("VACUUM")
    after = os.path.getsize(DB_PATH)
    saved = before - after
    if saved > 0:
        print(f"    VACUUM: {before/1024/1024:.1f} MB - {after/1024/1024:.1f} MB (-{saved/1024:.0f} KB)")
    else:
        print(f"    VACUUM: {after/1024/1024:.1f} MB (без изменений)")
    return saved


# -- Вывод --------------------------------------------------------------------

def print_stats(cur, tables):
    """Статистика по таблицам."""
    print(f"\n  {'Table':<25} {'Rows':>8}  {'Min Date':<12} {'Max Date':<12}")
    print(f"  {'-'*58}")
    for t in tables:
        row = cur.execute(
            f"SELECT COUNT(*), MIN(Date), MAX(Date) FROM {t}"
        ).fetchone()
        cnt, d_min, d_max = row
        print(f"  {t:<25} {cnt:>8}  {d_min or '-':<12} {d_max or '-':<12}")


def run_diagnostics(cur, tables):
    """Запускает все проверки, возвращает dict результатов."""
    W = 60
    print()
    print("=" * W)
    print(f"  DB CHECK  |  market.db  |  {len(tables)} tables")
    print("=" * W)

    results = {}

    # 1
    print("\n  [1/5] Duplicates by Date")
    dups = check_duplicates(cur, tables)
    results["dups"] = dups
    if dups:
        total_extra = sum(v["extra_rows"] for v in dups.values())
        print(f"  [!] Found in {len(dups)} tables  (+{total_extra} extra rows)")
        for t, v in sorted(dups.items()):
            print(f"       {t:<28} {v['dates']} dates  +{v['extra_rows']} rows")
    else:
        print("  [OK] No duplicates")

    # 2
    print("\n  [2/5] Date format (YYYY-MM-DD)")
    bad_fmt = check_date_formats(cur, tables)
    results["bad_fmt"] = bad_fmt
    if bad_fmt:
        print(f"  [!] Found in {len(bad_fmt)} tables")
        for t, v in sorted(bad_fmt.items()):
            print(f"       {t:<28} {v['count']} rows  e.g.: {v['examples'][:2]}")
    else:
        print("  [OK] All dates in correct format")

    # 3
    print("\n  [3/5] Hidden duplicates (mixed date formats)")
    hidden = check_hidden_duplicates(cur, tables)
    results["hidden"] = hidden
    if hidden:
        print(f"  [!] Found in {len(hidden)} tables")
        for t, cnt in sorted(hidden.items()):
            print(f"       {t:<28} {cnt} dates")
    else:
        print("  [OK] No hidden duplicates")

    # 4
    print("\n  [4/5] NULL values (Date, Close columns)")
    nulls = check_nulls(cur, tables)
    results["nulls"] = nulls
    if nulls:
        print(f"  [!] Found in {len(nulls)} tables")
        for t, cnt in sorted(nulls.items()):
            print(f"       {t:<28} {cnt} rows")
    else:
        print("  [OK] No NULL values")

    # 5
    print("\n  [5/5] Empty tables")
    empty = check_empty_tables(cur, tables)
    results["empty"] = empty
    if empty:
        print(f"  [!] {len(empty)} empty: {', '.join(empty)}")
    else:
        print("  [OK] No empty tables")

    results["has_problems"] = bool(dups or bad_fmt or hidden or nulls)
    return results


def run_quality_audit(conn, cur, tables):
    """Read-only блок аудита качества данных. Возвращает число предупреждений."""
    W = 60
    print(f"\n{'='*W}")
    print("  DATA QUALITY  |  read-only (--fix это не чинит)")
    print("=" * W)
    warn = 0
    price = _price_tables(cur, tables)  # OHLCV-таблицы, без лог-таблиц

    print("\n  [+] Integrity (PRAGMA quick_check)")
    integ = check_integrity(conn)
    if integ["quick_check"] == "ok":
        print(f"  [OK] quick_check ok  |  {integ['size_mb']:.1f} MB  |  {len(price)} price tables")
    else:
        print(f"  [!] quick_check: {integ['quick_check']}"); warn += 1

    print("\n  [+] Freshness (stale tables)")
    stale = check_freshness(cur, price)
    if stale:
        warn += len(stale)
        print(f"  [!] {len(stale)} tables stale "
              f"(daily >{STALE_DAILY_DAYS}d / weekly >{STALE_WEEKLY_DAYS}d):")
        for t, age in list(stale.items())[:20]:
            print(f"       {t:<28} {age} days old")
        if len(stale) > 20:
            print(f"       ... +{len(stale) - 20} more")
    else:
        print("  [OK] all tables fresh")

    print("\n  [+] OHLC integrity")
    ohlc = check_ohlc(cur, price)
    crit = {t: v["critical"] for t, v in ohlc.items() if v["critical"]}
    minor = {t: v["minor"] for t, v in ohlc.items() if v["minor"] and not v["critical"]}
    if crit:
        warn += len(crit)
        print(f"  [!] CRITICAL in {len(crit)} tables (High<Low / price<=0 / volume<0):")
        for t, c in sorted(crit.items()):
            print(f"       {t:<28} {c} rows")
    else:
        print("  [OK] no critical OHLC corruption")
    if minor:
        rows = sum(minor.values())
        print(f"  [..] minor: {len(minor)} tables, {rows} rows with Open/Close just "
              "outside [Low,High] (Yahoo FX feed quirk, not corruption)")

    print("\n  [+] Date gaps (missing daily bars)")
    gaps = check_gaps(cur, price)
    if gaps:
        warn += len(gaps)
        print(f"  [!] {len(gaps)} tables with a gap > {MAX_GAP_DAYS} days:")
        for t, (g, span) in list(gaps.items())[:15]:
            where = f" ({span[0]}..{span[1]})" if span else ""
            print(f"       {t:<28} {g} days{where}")
        if len(gaps) > 15:
            print(f"       ... +{len(gaps) - 15} more")
    else:
        print("  [OK] no large gaps")

    print(f"\n  [+] Low data (< {MIN_ROWS_TRAIN} rows, weak for training)")
    low = check_low_data(cur, price)
    if low:
        warn += len(low)
        print(f"  [!] {len(low)} thin tables: " +
              ", ".join(f"{t}({c})" for t, c in low.items()))
    else:
        print("  [OK] all daily tables have enough history")

    print("\n  [+] Registry coverage (config.FULL_ASSET_MAP)")
    cov = check_coverage(tables)
    if cov is None:
        print("  [..] config not importable - skipped")
    else:
        if cov["missing"]:
            warn += len(cov["missing"])
            print(f"  [!] {len(cov['missing'])} expected tables MISSING: "
                  + ", ".join(cov["missing"]))
        if cov["no_weekly"]:
            print(f"  [..] {len(cov['no_weekly'])} assets without a _weekly table: "
                  + ", ".join(cov["no_weekly"][:20])
                  + (" ..." if len(cov["no_weekly"]) > 20 else ""))
        if cov["orphan"]:
            print(f"  [..] {len(cov['orphan'])} tables not in config (orphans): "
                  + ", ".join(cov["orphan"]))
        if not cov["missing"] and not cov["orphan"] and not cov["no_weekly"]:
            print("  [OK] tables match the asset registry")

    print(f"\n  Data-quality warnings: {warn}")
    return warn


def run_fix(conn, cur, tables, results):
    """Исправляет все найденные проблемы."""
    print()
    print("=" * 60)
    print("  DB FIX  |  Auto-repair")
    print("=" * 60)

    fixed_total = 0

    if results.get("bad_fmt"):
        print("\n  [FIX] Нормализация дат:")
        fixed_total += fix_date_formats(cur, tables)

    if results.get("dups") or results.get("hidden"):
        print("\n  [FIX] Удаление дубликатов:")
        fixed_total += fix_duplicates(cur, tables)

    if results.get("nulls"):
        print("\n  [FIX] Удаление NULL-строк:")
        fixed_total += fix_nulls(cur, tables)

    conn.commit()

    print("\n  [FIX] Сжатие БД:")
    fix_vacuum(conn)

    # Перепроверка
    print(f"\n{'='*60}")
    print("  ПЕРЕПРОВЕРКА")
    print(f"{'='*60}")
    dups2 = check_duplicates(cur, tables)
    bad2 = check_date_formats(cur, tables)
    hidden2 = check_hidden_duplicates(cur, tables)
    nulls2 = check_nulls(cur, tables)

    if not dups2 and not bad2 and not hidden2 and not nulls2:
        print(f"\n  ГОТОВО: исправлено {fixed_total} строк. База чистая.")
    else:
        remaining = len(dups2) + len(bad2) + len(hidden2) + len(nulls2)
        print(f"\n  ВНИМАНИЕ: осталось {remaining} проблем. Запустите ещё раз.")

    return fixed_total


# -- Точки входа --------------------------------------------------------------

def main_audit():
    """Полный read-only аудит. Ничего не меняет (безопасно из лаунчера)."""
    conn = _connect()
    cur = conn.cursor()
    tables = get_tables(cur)

    results = run_diagnostics(cur, tables)
    warn = run_quality_audit(conn, cur, tables)

    print(f"\n{'='*60}")
    if not results["has_problems"] and not warn:
        print("  РЕЗУЛЬТАТ: база чистая, замечаний нет")
    else:
        fixable = "есть" if results["has_problems"] else "нет"
        print(f"  РЕЗУЛЬТАТ: исправимых проблем - {fixable}; "
              f"предупреждений по качеству - {warn}")
        if results["has_problems"]:
            print("  Запусти  python db_check.py --fix  для авто-ремонта.")
    print(f"{'='*60}")
    print_stats(cur, tables)
    conn.close()


def main_fix():
    """Аудит + авто-ремонт исправимых проблем (без вопросов)."""
    conn = _connect()
    cur = conn.cursor()
    tables = get_tables(cur)

    results = run_diagnostics(cur, tables)

    if results["has_problems"]:
        run_fix(conn, cur, tables, results)
    else:
        print("\n  Исправимых проблем нет.")

    run_quality_audit(conn, cur, tables)
    print_stats(cur, tables)
    conn.close()


def main_stats():
    """Только статистика."""
    conn = _connect()
    cur = conn.cursor()
    tables = get_tables(cur)
    print(f"\n  DB STATS - market.db ({len(tables)} таблиц)")
    print_stats(cur, tables)
    conn.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DB Audit & Fix для market.db")
    parser.add_argument("--fix", action="store_true", help="Авто-ремонт исправимых проблем")
    parser.add_argument("--stats", action="store_true", help="Только статистика")
    # Обратная совместимость
    parser.add_argument("--audit", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--autofix", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.stats:
        main_stats()
    elif args.fix or args.autofix:
        main_fix()
    else:
        main_audit()
