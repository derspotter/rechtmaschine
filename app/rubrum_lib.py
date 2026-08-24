#!/usr/bin/env python3
"""rubrum-cli — Gerichtsrubren für Schriftsätze bauen und Kanzlei-Formatierung erzwingen.

Zwei Kernprobleme, die dieses Tool reproduzierbar löst (bisher Handarbeit pro Dokument):

1. j-lawyer-Vorlagen rendern nur EINEN Mandanten ins Rubrum. `patch` ersetzt den
   Rubrum-Block eines gerenderten ODT durch beliebig viele Parteien aus einer
   JSON-Spezifikation (alle Rechtsbehelfe, alle Gerichtsbarkeiten).
2. Kanzlei-Formatierung (Jay): Rollenbezeichnungen ("Kläger,", "Beklagte,",
   "Antragsteller,", …) stehen RECHTSBÜNDIG in eigener Zeile, nummerierte
   Anträge sind EINGERÜCKT (0.5in). `format` erzwingt beides in jedem ODT.
   Achtung: automatische Stilnamen (P6/P7/…) sind pro Dokument vergeben —
   dieses Tool injiziert deshalb eigene Stile mit den im Dokument tatsächlich
   verwendeten Namespace-Präfixen, statt Stilnamen zu raten.

Kein LibreOffice nötig, reine Stdlib (zipfile + Regex auf content.xml).
"""

import argparse
import json
import re
import shutil
import sys
import zipfile
from collections import Counter

KANZLEI = "Kanzlei Keienborg, Friedrich-Ebert-Straße 17, 40210 Düsseldorf"

# Rollenzeilen, die rechtsbündig gehören (Zeile besteht exakt aus Rolle + Komma).
ROLLEN = (
    "Kläger", "Klägerin", "Klägerinnen", "Beklagte", "Beklagter",
    "Antragsteller", "Antragstellerin", "Antragsgegner", "Antragsgegnerin",
    "Widerspruchsführer", "Widerspruchsführerin", "Widerspruchsgegnerin",
    "Berufungskläger", "Berufungsklägerin", "Berufungsbeklagte",
    "Beschwerdeführer", "Beschwerdeführerin", "Beschwerdegegner", "Beschwerdegegnerin",
    "Beigeladene", "Beigeladener", "Beteiligte", "Beteiligter",
    "Vollstreckungsgläubiger", "Vollstreckungsschuldnerin", "Vollstreckungsschuldner",
)
# Genitiv ("Klägers,") und kombinierte Rollen ("Beklagte und Antragsgegnerin,")
# sind im Vorbild ebenfalls rechtsbündig (SKILL.md, Kopf und Rubrum).
_ROLLE_ALT = r"(?:" + "|".join(ROLLEN) + r")s?"
ROLLEN_RE = re.compile(r"^" + _ROLLE_ALT + r"(?: und " + _ROLLE_ALT + r")?\s*,?$")

# Lead-in-Zeilen, nach denen nummerierte Absätze als Anträge gelten.
ANTRAG_START_RE = re.compile(
    r"(beantrage[n]?\s+(ich|wir)"          # beantrage ich / beantragen wir
    r"|erhebe[n]?\s+(ich|wir)\b.{0,120}?\bund\s+beantrage"  # erhebe ich ... Klage und beantrage,
    r"|beantrag(?:e[n]?|t)\s*[,:]\s*$"     # Absatz endet auf 'beantrage,' / 'beantragt:'
    r"|folgende[n]?\s+Anträge"
    r"|forder(?:e|n)?\s+(ich|wir)\s+(Sie|sie)\b.{0,120}?\bauf\s*[,:]\s*$"  # fordere ich Sie (daher) auf, — außergerichtliche Aufforderung (Brill 134/26, 21.07.2026)
    r"|wird beantragt\s*[,:]?\s*$)", re.IGNORECASE)
ANTRAG_ENDE_RE = re.compile(r"^\s*(Begründung|Gründe)\s*:?\s*$")
NUMMERIERT_RE = re.compile(r"^\s*\d+\.\s")

# Ein Absatz ist ein ECHTER Antrag-Lead-in, wenn er auf ','/':' endet ODER auf
# die Lead-in-Verbphrase selbst ("… beantrage ich" / "… bitte ich um" —
# Marcels 25a-Briefe enden ohne Komma), optional gefolgt von EINEM Adverb aus
# der Whitelist ("… beantragen wir zunächst" — 137/26, 19.07.2026: das
# nachgestellte 'zunächst' ließ die Antragszone nicht öffnen, 'Akteneinsicht'
# blieb unformatiert und check meldete nichts). Bewusst KEINE beliebigen
# Trailing-Wörter: inline erledigte Anträge ("Zugleich beantragen wir
# Einsicht … .") dürfen weiterhin nicht zählen.
_LEADIN_ADVERB = r"(?:\s+(?:zunächst|außerdem|ferner|zudem|weiter|zusätzlich|insoweit|hilfsweise))?"
# Bei "bitte(n) ich/wir um" steht das Adverb ZWISCHEN Verb und "um"
# ("Insoweit bitten wir außerdem um / Akteneinsicht" — 152/26, 31.07.2026:
# die Zone öffnete nicht, 'Akteneinsicht' blieb unformatiert, check OK).
_BITTE_UM_RE = r"bitte[n]?\s+(ich|wir)" + _LEADIN_ADVERB + r"\s+um\s*$"
LEADIN_ENDE_RE = re.compile(
    r"([,:]\s*$"
    r"|beantrag(e[n]?|t)\s+(ich|wir)" + _LEADIN_ADVERB + r"\s*$"
    r"|" + _BITTE_UM_RE + r")",
    re.IGNORECASE)


def ist_leadin(text):
    return bool((ANTRAG_START_RE.search(text) or
                 re.search(_BITTE_UM_RE, text, re.IGNORECASE))
                and LEADIN_ENDE_RE.search(text))


# Nummerierte AUFZÄHLUNGEN (keine Anträge) werden eingerückt, aber NICHT fett
# (Jay, 28.07.2026, 003/26 Herzquartier-Schreiben: die Liste der nach § 60a
# Abs. 2c AufenthG geforderten Attest-Angaben blieb unformatiert, weil der
# Lead-in "… sind insbesondere folgende Angaben erforderlich:" kein
# Antrags-Lead-in ist — die Zustandsmaschine öffnete keine Zone und `check`
# meldete OK). Erkennung rein STRUKTURELL an der Nummerierung nach einem
# doppelpunkt-beendeten Lead-in, nicht semantisch: so greift die Regel auch
# dann, wenn beim Verfassen niemand an sie denkt.
AUFZAEHLUNG_LEADIN_RE = re.compile(r":\s*$")
# Gliederungs-Doppelpunkte öffnen KEINE Aufzählungszone: nach "Begründung:" bzw.
# "Zur Begründung:" folgen Abschnitte, keine eingerückte Liste — sonst würden
# arabisch nummerierte Gliederungspunkte fälschlich eingerückt.
_KEIN_AUFZAEHLUNG_LEADIN_RE = re.compile(
    r"^\s*(Begründung|Gründe|Zur Begründung|Sachverhalt|Anlagen?)\s*:?\s*$")


def _kein_aufzaehlung_leadin(text):
    return bool(_KEIN_AUFZAEHLUNG_LEADIN_RE.match(text))


# Akteneinsicht-Anträge werden zentriert, alle übrigen eingerückt
# (Jay, 14.07.2026: "Akteneinsicht zentriert, Rest eingerückt").
AKTENEINSICHT_RE = re.compile(r"^(Akteneinsicht|Einsicht in die)")

# Weitere Konventionsmuster (Formatvorbild Keienborg, siehe SKILL.md).
VERFAHREN_LEADIN_RE = re.compile(
    r"^In de(m|r)\s.{0,80}([Vv]erfahren|Rechtsstreit|Strafsache|Verwaltungsstreitsache)$")  # auch Komposita: "einstweiligen Rechtsschutzverfahren"
AZ_RE = re.compile(
    r"^(\d+\s+[A-Za-z]{1,4}\s+\d+/\d{2}(\.[A-Z])?"   # 41 K 2206/26.A / 17 B 814/17
    r"|[A-Z]{1,3}\s+\d+\s+[A-Za-z]{1,4}\s+\d+/\d{2}(\s+[A-Z]{1,3}){0,2}"  # S 28 AY 45/26 ER (SG/LSG)
    r"|Az\.?\s*:.{1,60})\s*$")
HILFSWEISE_RE = re.compile(r"^(weiter\s+)?hilfsweise\s*,?\s*$", re.IGNORECASE)
BEGRUENDUNG_RE = re.compile(r"^(Begründung|Gründe)\s*:?\s*$")
# Behördenbriefe: "Zur Begründung:" als eigener Absatz wird zentriert, NICHT
# fett (Formatvorbild-Ergänzung 19.07.2026, empirisch aus 137/26: align=center,
# kein font-weight). Nur im --behoerde-Modus; Gericht behält "Begründung:"
# fett+unterstrichen (BEGRUENDUNG_TITEL).
ZUR_BEGRUENDUNG_RE = re.compile(r"^Zur Begründung\s*:?\s*$")
# Fristsetzung als eigener Absatz (nur das Datum) wird fett+zentriert
# (Formatvorbild "Hervorgehobene Einzelanträge im Fließtext": "10. Juli 2025").
# Optionaler Satzpunkt: endet der umbrochene Satz auf dem Datum ("… bitten wir
# bis zum / 05.08.2026."), blieb der Absatz unerkannt (152/26, 31.07.2026).
FRIST_DATUM_RE = re.compile(
    r"^(\d{1,2}\.\s?\d{1,2}\.\s?\d{4}"
    r"|\d{1,2}\.\s?(Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\s+\d{4})\s*\.?\s*$")
ABSCHNITT_RE = re.compile(r"^[IVX]{1,5}\.\s+\S")
# Beweisangebote/Glaubhaftmachungen und nummerierte Anlagen-Referenzen als
# eigener Absatz sind fett + eingerückt (Empirie 28.07.2026: 6/6 Keienborg/
# Schotte-Klagebegründungen 2024–2025 mit Beweisangeboten — "Beweis: Kopie
# des …, Anlage K4" durchgehend ml=1.27cm, bold; 021/25, 085/25, 119/25,
# 149/25, 169/25, 233/24). Die nackte Anlagen-Zeile unter der Signatur
# (ANLAGE_RE) ist davon unberührt.
# Anlagen-Nummern auch mit Parteikürzel (K 1, AS 2, B 3 — "Anlage AS 2: …",
# Eilantrag 125/26, 11.08.2026: "Anlage AS n:"-Zeilen blieben unformatiert).
BEWEIS_RE = re.compile(r"^(Beweis|Glaubhaftmachung)\s*:|^Anlage\s*(?:[A-Z]{1,3}\s*)?\d+\s*:")
ANLAGE_RE = re.compile(r"^Anlage(n)?\s*$")
# Erwähnt der Fließtext irgendeine Anlage? (auch "beigefügt"/"anbei" ohne das
# Wort Anlage — "Ausweislich der beigefügten Vollmacht …")
ANLAGE_REF_RE = re.compile(r"Anlage|beigefügt|beifüg|fügen wir|füge ich|anbei|anliegend",
                           re.IGNORECASE)
GRUSS_RE = re.compile(r"^(Mit freundlichen Grüßen|Hochachtungsvoll)\b")
PV_RE = re.compile(r"^-\s*Prozessbevollmächtigte[rn]?\s*:")
AZ_LEER_RE = re.compile(r"^Az\.?\s*:\s*$")


def _tidy_spaces(par_xml):
    """Platzhalter-Artefakte bereinigen: Doppel-Leerzeichen und Leerzeichen vor
    Satzzeichen — auch ÜBER Span-Grenzen hinweg (j-lawyer rendert Platzhalter
    als eigene Spans mit angehängten Leerzeichen). Tags bleiben unberührt."""
    teile = re.split(r"(<[^>]+>)", par_xml)
    text_idx = [i for i, x in enumerate(teile) if x and not x.startswith("<")]
    for i in text_idx:
        teile[i] = re.sub(r"  +", " ", teile[i])
        # " ./." (Kurzrubrum-Separator) ist KEIN Leerzeichen-Artefakt
        teile[i] = re.sub(r" +([,.;:!?])(?!/\.)", r"\1", teile[i])
    prev = None
    for i in text_idx:
        seg = teile[i]
        if not seg:
            continue
        if prev is not None:
            if seg[0] in ",.;:!?" and not seg.startswith("./."):
                teile[prev] = teile[prev].rstrip(" ")
            elif seg.startswith(" ") and teile[prev].endswith(" "):
                seg = teile[i] = seg.lstrip(" ")
        if teile[i]:
            prev = i
    # Leerzeichen am Absatz-ENDE entfernen (j-lawyer hängt an jeden
    # aufgelösten Platzhalter ein Leerzeichen — 'Az.: 33.61.02/191327 ',
    # 137/26, 19.07.2026): letztes Textstück rstrippen.
    for i in reversed(text_idx):
        if teile[i]:
            teile[i] = teile[i].rstrip()
            break
    return "".join(teile)
ROEM_ARABISCH_HEADING_RE = ABSCHNITT_RE  # Alias


def ist_unterstrichene_ueberschrift(text):
    """Absätze, die als UNTERSTRICHENE Überschrift formatiert werden.

    Das sind genau zwei: "Begründung:"/"Gründe:" (BEGRUENDUNG_TITEL, fett +
    unterstrichen) und die Abschnittsüberschriften "I. …"/"II. …"
    (ABSCHNITT_TITEL, zentriert + unterstrichen). "Zur Begründung:" gehört
    NICHT dazu — der Absatz ist zentriert, aber nicht unterstrichen
    (BEGRUENDUNG_ZENTRIERT).

    Vor jeder solchen Überschrift steht eine Leerzeile (Jay, 11.08.2026).
    """
    return bool(BEGRUENDUNG_RE.match(text)
                or (ABSCHNITT_RE.match(text) and len(text) < 60))


def _read(path):
    src = zipfile.ZipFile(path)
    return src, src.read("content.xml").decode("utf-8")


def _write(src, xml, out_path):
    out = zipfile.ZipFile(out_path, "w")
    for item in src.infolist():
        data = xml.encode("utf-8") if item.filename == "content.xml" else src.read(item.filename)
        comp = zipfile.ZIP_STORED if item.filename == "mimetype" else zipfile.ZIP_DEFLATED
        out.writestr(item, data, comp)
    out.close()


def _prefixes(xml):
    """Namespace-Präfixe des Dokuments erkennen (LibreOffice: style/text/fo,
    j-lawyer fix-ingo-text: ns0/ns1/ns5/ns7 o.ä.)."""
    m_styles = re.search(r"<(\w+):automatic-styles[\s>]", xml)
    m_styledef = re.search(r"<(\w+):style [^>]*\1:family=\"paragraph\"", xml)
    m_para = re.search(r"<(\w+):p [^>]*style-name=", xml)
    # fo-Präfix: der, der bei text-align/margin-left verwendet wird — aus einer
    # vorhandenen paragraph-properties ableiten, sonst Standard 'fo'.
    m_fo = re.search(r"<\w+:paragraph-properties [^>]*?(\w+):(?:text-align|line-height|margin-left)=", xml)
    return {
        "auto": m_styles.group(1) if m_styles else "office",
        "style": m_styledef.group(1) if m_styledef else "style",
        "text": m_para.group(1) if m_para else "text",
        "fo": m_fo.group(1) if m_fo else "fo",
    }


def _para_text(inner):
    return re.sub(r"<[^>]+>", "", inner).strip()


def _dominant_body_parent(xml, ns):
    """Benannten Elternstil des dominanten Fließtext-Absatzstils ermitteln.

    Der Parent der injizierten Stile entscheidet über die geerbte Schrift.
    Der erstbeste parent-style-name im Dokument zeigt je nach Vorlage auf
    eine Stilkette OHNE Schriftangabe (z.B. Text_20_body) — dann fallen
    Anträge/Überschriften auf die Dokument-Grundschrift zurück, während der
    Fließtext in der Vorlagenschrift steht (Empirie 31.07.2026, 152/26).
    Deshalb: den am häufigsten verwendeten Stil ECHTER Textabsätze (>60
    Zeichen, keine eigenen [A-Z_]-Stile) nehmen und dessen benannten Parent
    zurückgeben (Autostil → sein Parent, benannter Stil → er selbst).
    """
    t, s = ns["text"], ns["style"]
    counts = Counter()
    for m in re.finditer(r"<" + t + r":p [^>]*" + t + r":style-name=\"([^\"]+)\"[^>]*>(.*?)</" + t + r":p>", xml, re.S):
        name, inner = m.group(1), m.group(2)
        # Nur eigene injizierte Stile überspringen (GROSS_MIT_UNTERSTRICH) —
        # ein Muster wie [A-Z0-9_]+ träfe auch die Autostile P4/P14.
        if re.fullmatch(r"[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+", name):
            continue
        if len(_para_text(inner)) > 60:
            counts[name] += 1
    for name, _ in counts.most_common():
        m = re.search(r"<" + s + r":style [^>]*" + s + r":name=\"" + re.escape(name) + r"\"[^>]*" + s + r":parent-style-name=\"([^\"]+)\"", xml)
        if m:
            return m.group(1)
        if not re.search(r"<" + s + r":style [^>]*" + s + r":name=\"" + re.escape(name) + r"\"", xml):
            return name  # benannter Stil aus styles.xml, direkt verwendbar
    return None


def _ensure_styles(xml, ns):
    """Rechtsbündig- und Einzug-Stil injizieren, falls nicht vorhanden."""
    s, fo = ns["style"], ns["fo"]
    parent = _dominant_body_parent(xml, ns)
    if not parent:
        parent = "body"
        m = re.search(r"<" + s + r":style [^>]*" + s + r":parent-style-name=\"([^\"]+)\"", xml)
        if m:
            parent = m.group(1)
    if True:  # fehlende Stile einzeln injizieren (idempotent, s.u.)
        neu = (
            f'<{s}:style {s}:family="paragraph" {s}:name="RUBRUM_BODY" {s}:parent-style-name="{parent}">'
            f'<{s}:paragraph-properties {fo}:text-align="justify" {s}:justify-single-word="false" '
            f'{fo}:margin-left="0cm" {fo}:text-indent="0cm" {s}:auto-text-indent="false" /></{s}:style>'
            f'<{s}:style {s}:family="paragraph" {s}:name="PV_EINZUG" {s}:parent-style-name="{parent}">'
            f'<{s}:paragraph-properties {fo}:text-align="start" {fo}:margin-left="6.35cm" '
            f'{fo}:text-indent="0cm" {s}:auto-text-indent="false" /></{s}:style>'
            f'<{s}:style {s}:family="paragraph" {s}:name="PV_ZEILE1" {s}:parent-style-name="{parent}">'
            f'<{s}:paragraph-properties {fo}:text-align="start" {fo}:margin-left="0cm" '
            f'{fo}:text-indent="0cm" {s}:auto-text-indent="false">'
            f'<{s}:tab-stops><{s}:tab-stop {s}:position="6.35cm" /></{s}:tab-stops>'
            f'</{s}:paragraph-properties></{s}:style>'
            f'<{s}:style {s}:family="paragraph" {s}:name="RUBRUM_RECHTS" {s}:parent-style-name="{parent}">'
            f'<{s}:paragraph-properties {fo}:text-align="right" {s}:justify-single-word="false" /></{s}:style>'
            f'<{s}:style {s}:family="paragraph" {s}:name="ANTRAG_EINZUG" {s}:parent-style-name="{parent}">'
            f'<{s}:paragraph-properties {fo}:margin-left="0.5in" {fo}:text-align="justify" '
            f'{s}:justify-single-word="false" {fo}:text-indent="0in" {s}:auto-text-indent="false" /></{s}:style>'
            f'<{s}:style {s}:family="paragraph" {s}:name="AUFZAEHLUNG_EINZUG" {s}:parent-style-name="{parent}">'
            f'<{s}:paragraph-properties {fo}:margin-left="0.5in" {fo}:text-align="justify" '
            f'{s}:justify-single-word="false" {fo}:text-indent="0in" {s}:auto-text-indent="false" /></{s}:style>'
            f'<{s}:style {s}:family="paragraph" {s}:name="ANTRAG_FETT" {s}:parent-style-name="{parent}">'
            f'<{s}:paragraph-properties {fo}:margin-left="0.5in" {fo}:text-align="justify" '
            f'{s}:justify-single-word="false" {fo}:text-indent="0in" {s}:auto-text-indent="false" />'
            f'<{s}:text-properties {fo}:font-weight="bold" {s}:font-weight-asian="bold" {s}:font-weight-complex="bold" /></{s}:style>'
            f'<{s}:style {s}:family="paragraph" {s}:name="RUBRUM_TITEL" {s}:parent-style-name="{parent}">'
            f'<{s}:paragraph-properties {fo}:text-align="center" {s}:justify-single-word="false" />'
            f'<{s}:text-properties {fo}:font-size="14pt" {fo}:font-weight="bold" {s}:font-weight-asian="bold" {s}:font-weight-complex="bold" /></{s}:style>'
            f'<{s}:style {s}:family="paragraph" {s}:name="RUBRUM_WEGEN" {s}:parent-style-name="{parent}">'
            f'<{s}:text-properties {fo}:font-weight="bold" {s}:font-weight-asian="bold" {s}:font-weight-complex="bold" /></{s}:style>'
            f'<{s}:style {s}:family="paragraph" {s}:name="PARTEI_FETT" {s}:parent-style-name="{parent}">'
            f'<{s}:paragraph-properties {fo}:text-align="justify" {s}:justify-single-word="false" />'
            f'<{s}:text-properties {fo}:font-weight="bold" {s}:font-weight-asian="bold" {s}:font-weight-complex="bold" /></{s}:style>'
            f'<{s}:style {s}:family="paragraph" {s}:name="AZ_UNTERSTRICHEN" {s}:parent-style-name="{parent}">'
            f'<{s}:text-properties {s}:text-underline-style="solid" {s}:text-underline-width="auto" '
            f'{s}:text-underline-color="font-color" /></{s}:style>'
            f'<{s}:style {s}:family="paragraph" {s}:name="BEGRUENDUNG_TITEL" {s}:parent-style-name="{parent}">'
            f'<{s}:text-properties {fo}:font-weight="bold" {s}:font-weight-asian="bold" {s}:font-weight-complex="bold" '
            f'{s}:text-underline-style="solid" {s}:text-underline-width="auto" {s}:text-underline-color="font-color" /></{s}:style>'
            f'<{s}:style {s}:family="paragraph" {s}:name="ABSCHNITT_TITEL" {s}:parent-style-name="{parent}">'
            f'<{s}:paragraph-properties {fo}:text-align="center" {s}:justify-single-word="false" />'
            f'<{s}:text-properties {s}:text-underline-style="solid" {s}:text-underline-width="auto" '
            f'{s}:text-underline-color="font-color" /></{s}:style>'
            f'<{s}:style {s}:family="paragraph" {s}:name="ANTRAG_ZENTRIERT" {s}:parent-style-name="{parent}">'
            f'<{s}:paragraph-properties {fo}:text-align="center" {s}:justify-single-word="false" />'
            f'<{s}:text-properties {fo}:font-weight="bold" {s}:font-weight-asian="bold" {s}:font-weight-complex="bold" /></{s}:style>'
            f'<{s}:style {s}:family="paragraph" {s}:name="BEGRUENDUNG_ZENTRIERT" {s}:parent-style-name="{parent}">'
            f'<{s}:paragraph-properties {fo}:text-align="center" {s}:justify-single-word="false" /></{s}:style>'
        )
        # Nur die Stile injizieren, die noch fehlen (idempotent bei Wiederholung).
        fehlend = "".join(
            frag for frag in re.findall(r"<" + s + r":style .*?</" + s + r":style>", neu)
            if re.search(s + r':name="([A-Z0-9_]+)"', frag).group(1) not in xml
        )
        if fehlend:
            close = f"</{ns['auto']}:automatic-styles>"
            if close not in xml:
                raise SystemExit(f"automatic-styles ({close}) nicht gefunden")
            xml = xml.replace(close, fehlend + close, 1)
        # Selbstheilung: früher injizierte Stile mit abweichendem (falschem)
        # Parent auf den ermittelten Body-Parent umhängen — sonst bleibt ein
        # einmal falsch geparentes Dokument trotz erneutem format falsch.
        for name in re.findall(s + r':name="([A-Z0-9_]+)"', neu):
            xml = re.sub(
                r"(<" + s + r":style [^>]*" + s + r':name="' + name + r'"[^>]*' + s + r':parent-style-name=")[^"]+(")',
                lambda m: m.group(1) + parent + m.group(2),
                xml,
            )
    return xml


def _set_style(par_full, t, style_name):
    return re.sub(t + r':style-name="[^"]+"', t + f':style-name="{style_name}"', par_full, count=1)


HEADING_RE = re.compile(
    r"^(Klage(\s+und\s+Antrag\s+(gemäß|nach)\s+§\s*80\s*Abs\.?\s*5\s*VwGO)?"
    r"|K\s*l\s*a\s*g\s*e|Antrag (auf|gemäß|nach) .{0,60}"
    r"|Widerspruch(?: gegen .{0,80})?|Berufung|Beschwerde(?:begründung)?)$"
)
WEGEN_RE = re.compile(r"^wegen .{1,80}$", re.IGNORECASE)


def format_file(path, out_path=None, behoerde=False):
    """Convenience: ODT-Datei einlesen, format_odt anwenden, zurückschreiben."""
    src, xml = _read(path)
    ns = _prefixes(xml)
    xml = format_odt(xml, ns, behoerde=behoerde)
    target = out_path or path
    if target == path:
        tmp = path + ".tmp"
        _write(src, xml, tmp)
        src.close()
        shutil.move(tmp, path)
    else:
        _write(src, xml, target)
        src.close()
    return target


def format_odt(xml, ns, behoerde=False):
    """Kanzlei-Konventionen erzwingen (Formatvorbild Keienborg, siehe SKILL.md).

    Gericht: Überschrift fett+zentriert, Parteienzeile fett, Az. unterstrichen,
    Rollenzeilen rechtsbündig, nummerierte Anträge fett+eingerückt, 'hilfsweise'
    nicht fett, 'Begründung:' fett+unterstrichen, 'I. …' zentriert+unterstrichen.
    Mit behoerde=True: Antrag nach dem Lead-in fett+zentriert statt nummeriert.
    """
    xml = _ensure_styles(xml, ns)
    t = ns["text"]
    para_re = re.compile(r"<" + t + r":p [^>]*?(?<!/)>(.*?)</" + t + r":p>|<" + t + r":p [^>]*/>", re.DOTALL)
    out = []
    pos = 0
    in_antraege = False
    aufzaehlung_offen = False
    rubrum_state = None  # None | 'partei' | 'az'
    pv_folgezeile = False
    counts = {}

    def zaehl(k):
        counts[k] = counts.get(k, 0) + 1

    pending_leer = False
    last_was_leer = True
    leer_p = f'<{t}:p {t}:style-name="RUBRUM_BODY" />'
    alle_texte = [_para_text(m.group(1) or "") for m in para_re.finditer(xml)]
    anlagen_referenziert = any(ANLAGE_REF_RE.search(x) for x in alle_texte
                               if x and not ANLAGE_RE.match(x))
    for m in para_re.finditer(xml):
        text = _para_text(m.group(1) or "")
        par = m.group(0)
        neu = par
        if not text:
            pending_leer = False
            last_was_leer = True
            out.append(xml[pos:m.start()])
            out.append(neu)
            pos = m.end()
            continue
        if pending_leer:
            out.append(leer_p)
            zaehl("Leerzeile eingefügt")
            pending_leer = False
            last_was_leer = True
        if AZ_LEER_RE.match(text):
            # Leere "Az.:"-Zeile entfernen (Marcel lässt sie weg, wenn kein
            # Az. bekannt ist — 158/25).
            zaehl("leere Az.-Zeile entfernt")
            out.append(xml[pos:m.start()])
            pos = m.end()
            continue
        if ANLAGE_RE.match(text) and not anlagen_referenziert:
            # Vorlagen tragen die "Anlage(n)"-Zeile fest unter der Signatur —
            # ohne im Text referenzierte Anlage ist sie falsch (089/26, 17.07.2026).
            zaehl("Anlagenzeile ohne Anlagen entfernt")
            out.append(xml[pos:m.start()])
            pos = m.end()
            continue
        getidy = _tidy_spaces(par)
        if getidy != par:
            par = getidy
            neu = getidy
            zaehl("Leerzeichen bereinigt")
        if rubrum_state == "partei":
            neu = _set_style(par, t, "PARTEI_FETT")
            zaehl("Parteienzeile fett")
            rubrum_state = "az"
        elif rubrum_state == "az":
            if AZ_RE.match(text):
                neu = _set_style(par, t, "AZ_UNTERSTRICHEN")
                zaehl("Az. unterstrichen")
            rubrum_state = None
        elif ROLLEN_RE.match(text):
            neu = _set_style(par, t, "RUBRUM_RECHTS")
            zaehl("Rollenzeile rechtsbündig")
        elif PV_RE.match(text):
            if text.rstrip().endswith("-"):
                # Einzeilige Fassung aufspalten (Vorbild 011/26 Ahmadi):
                # Zeile 1 "- Prozessbevollmächtigte:<Tab>Kanzlei …," /
                # Zeile 2 (6.35cm Einzug) "Adresse -"
                rest = PV_RE.sub("", text).strip().rstrip("-").strip().rstrip(",")
                pv_name, _, pv_adresse = rest.partition(", ")
                if pv_adresse:
                    esc = lambda x: (x.replace("&", "&amp;")
                                     .replace("<", "&lt;").replace(">", "&gt;"))
                    neu = (
                        f'<{t}:p {t}:style-name="PV_ZEILE1">- Prozessbevollmächtigte:'
                        f'<{t}:tab/>{esc(pv_name)},</{t}:p>'
                        f'<{t}:p {t}:style-name="PV_EINZUG">{esc(pv_adresse)} -</{t}:p>'
                    )
                    zaehl("Prozessbevollmächtigte zweizeilig")
            else:
                pv_folgezeile = True
        elif pv_folgezeile:
            neu = _set_style(par, t, "PV_EINZUG")
            zaehl("PV-Adresszeile eingerückt")
            pv_folgezeile = False
        elif not in_antraege and HEADING_RE.match(text):
            neu = _set_style(par, t, "RUBRUM_TITEL")
            zaehl("Verfahrensüberschrift")
        elif WEGEN_RE.match(text):
            neu = _set_style(par, t, "RUBRUM_WEGEN")
            zaehl("wegen-Zeile")
        elif BEGRUENDUNG_RE.match(text):
            neu = _set_style(par, t, "BEGRUENDUNG_TITEL")
            zaehl("Begründung fett+unterstrichen")
        elif behoerde and ZUR_BEGRUENDUNG_RE.match(text):
            neu = _set_style(par, t, "BEGRUENDUNG_ZENTRIERT")
            zaehl("Zur Begründung zentriert")
        elif FRIST_DATUM_RE.match(text):
            neu = _set_style(par, t, "ANTRAG_ZENTRIERT")
            zaehl("Frist-Datum fett+zentriert")
        elif BEWEIS_RE.match(text):
            neu = _set_style(par, t, "ANTRAG_FETT")
            zaehl("Beweis/Anlagen-Referenz fett+eingerückt")
        elif ANLAGE_RE.match(text):
            # Referenzierte "Anlage(n)"-Zeile unter der Signatur: unterstrichen
            # (check verlangte das seit je, format setzte es nie — 161/25,
            # 24.08.2026; die unreferenzierte Zeile wird oben entfernt).
            neu = _set_style(par, t, "AZ_UNTERSTRICHEN")
            zaehl("Anlagenzeile unterstrichen")
        elif ABSCHNITT_RE.match(text) and len(text) < 60:
            neu = _set_style(par, t, "ABSCHNITT_TITEL")
            zaehl("Abschnittsüberschrift zentriert+unterstrichen")
        elif in_antraege and GRUSS_RE.match(text):
            in_antraege = False
        elif in_antraege and HILFSWEISE_RE.match(text):
            neu = _set_style(par, t, "ANTRAG_EINZUG")
            zaehl("hilfsweise (nicht fett)")
            in_antraege = "erster"
        elif in_antraege and NUMMERIERT_RE.match(text):
            neu = _set_style(par, t, "ANTRAG_FETT")
            zaehl("Antrag fett+eingerückt")
            in_antraege = "weiter"
        elif in_antraege == "erster":
            # Der (unnummerierte) Antrag direkt nach Lead-in bzw. 'hilfsweise,'
            # ist Antragstext. Akteneinsicht zentriert, alle übrigen eingerückt.
            if AKTENEINSICHT_RE.match(text):
                neu = _set_style(par, t, "ANTRAG_ZENTRIERT")
                zaehl("Akteneinsicht-Antrag fett+zentriert")
            else:
                neu = _set_style(par, t, "ANTRAG_FETT")
                zaehl("Antrag fett+eingerückt")
            in_antraege = "weiter"
        elif in_antraege == "weiter":
            # Nur 'hilfsweise,' oder nummerierte Absätze setzen die Zone fort.
            in_antraege = False
        elif aufzaehlung_offen and NUMMERIERT_RE.match(text):
            neu = _set_style(par, t, "AUFZAEHLUNG_EINZUG")
            zaehl("Aufzählungspunkt eingerückt")
        if VERFAHREN_LEADIN_RE.match(text):
            rubrum_state = "partei"
        lead_in = ist_leadin(text)
        if lead_in:
            in_antraege = "erster"
        # Aufzählungszone: öffnet nach einem doppelpunkt-beendeten Absatz, der
        # KEIN Antrags-Lead-in ist, und läuft solange nummerierte Absätze folgen.
        if lead_in or _kein_aufzaehlung_leadin(text):
            aufzaehlung_offen = False
        elif AUFZAEHLUNG_LEADIN_RE.search(text):
            aufzaehlung_offen = True
        elif not NUMMERIERT_RE.match(text):
            aufzaehlung_offen = False
        if (BEGRUENDUNG_RE.match(text) or ZUR_BEGRUENDUNG_RE.match(text)
                or (in_antraege and text.startswith("Begründung"))):
            in_antraege = False
        # Leerzeilen-Grammatik (Vorbild Keienborg 200420_vg_klage, 011/26):
        # nach 'wegen …', nach dem Lead-in und nach jedem Antragszonen-Element
        # folgt genau eine Leerzeile.
        if WEGEN_RE.match(text) or lead_in or (neu is not par and "ANTRAG_" in neu):
            pending_leer = True
        # Behördenbriefe (Jay-Ansage 14.07.2026 spät, ersetzt die Marcel-
        # Brief-Regel "Leerzeile nach jedem Absatz"): Fließtext läuft wie in
        # Gerichts-Schriftsätzen DIREKT untereinander (line-height 150% trägt
        # den Abstand). Erzwungen werden nur die Strukturgrenzen: Leerzeile
        # nach der Anrede und vor der Grußformel. Vorhandene Fließtext-
        # Leerzeilen (Marcel-Originale) bleiben unangetastet.
        if behoerde:
            if re.match(r"^Sehr geehrte", text):
                pending_leer = True
            # Strukturblöcke bekommen Leerzeile davor UND danach:
            # Abschnittsüberschriften (I. …), Zitatblöcke, Antrag-Lead-in.
            struktur = (
                (ABSCHNITT_RE.match(text) and len(text) < 60)
                or text.lstrip().startswith(("„", '"'))
                or lead_in
                or ZUR_BEGRUENDUNG_RE.match(text)
                or FRIST_DATUM_RE.match(text)
            )
            if struktur:
                if not last_was_leer:
                    out.append(leer_p)
                    zaehl("Leerzeile vor Strukturblock")
                    last_was_leer = True
                pending_leer = True
            if GRUSS_RE.match(text) and not last_was_leer:
                out.append(leer_p)
                zaehl("Leerzeile vor Grußformel")
        # Vor JEDER unterstrichenen Überschrift steht eine Leerzeile — in
        # Gerichts- wie in Behördendokumenten (Jay, 11.08.2026, an der
        # Untätigkeitsklage 114/26 gemerkt: "I. Zum Sachverhalt" klebte an
        # "Begründung:" und "IV. Akteneinsicht und Hinweise" am Fließtext
        # davor, während II. und III. eine Leerzeile hatten — der Body-Autor
        # setzt sie mal so, mal so). Läuft NACH dem behoerde-Block, der für
        # Abschnittsüberschriften schon last_was_leer setzt: keine Dopplung.
        # Greift nicht im Rubrum — dort ist keine Zeile eine Überschrift in
        # diesem Sinn, die Az.-Zeile hängt an rubrum_state.
        if ist_unterstrichene_ueberschrift(text) and not last_was_leer:
            out.append(leer_p)
            zaehl("Leerzeile vor unterstrichener Überschrift")
            last_was_leer = True
        last_was_leer = False
        out.append(xml[pos:m.start()])
        out.append(neu)
        pos = m.end()
    out.append(xml[pos:])
    zusammenfassung = ", ".join(f"{v}× {k}" for k, v in counts.items()) or "nichts zu tun"
    print(f"formatiert: {zusammenfassung}", file=sys.stderr)
    return "".join(out)


def build_rubrum_paragraphs(spec, ns, body_style="RUBRUM_BODY"):
    """Rubrum-Absätze aus der Spezifikation bauen (als content.xml-Fragment)."""
    t = ns["text"]

    def P(text, style=body_style):
        text = (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
        return f'<{t}:p {t}:style-name="{style}">{text}</{t}:p>'

    aktiv = spec["aktiv"]
    lines = []
    nummeriert = len(aktiv) > 1
    adressen = {p.get("adresse", "") for p in aktiv}
    gemeinsame_adresse = adressen.pop() if nummeriert and len(adressen) == 1 else None
    for i, p in enumerate(aktiv, 1):
        bez = p["bezeichnung"].rstrip(",")
        zusatz = f', geb. am {p["geb"]}' if p.get("geb") else ""
        # Mehrparteien mit gemeinsamer Adresse: Einzeladressen weglassen,
        # stattdessen EINE "alle wohnhaft:"-Zeile (Vorbild Eilantrag 096/26).
        adresse = "" if gemeinsame_adresse else (f', {p["adresse"]}' if p.get("adresse") else "")
        prefix = f"{i}. " if nummeriert else ""
        lines.append(P(f"{prefix}{bez}{zusatz}{adresse},"))
    zusaetze = spec.get("aktiv_zusatz", [])
    for z in zusaetze:
        lines.append(P(z if z.endswith(",") else z + ","))
    if gemeinsame_adresse and not any("wohnhaft" in z for z in zusaetze):
        lines.append(P(f"alle wohnhaft: {gemeinsame_adresse},"))
    lines.append(P(spec["aktiv_rolle"].rstrip(",") + ",", "RUBRUM_RECHTS"))
    pv = spec.get("prozessbevollmaechtigte", KANZLEI)
    pv_name, _, pv_adresse = pv.partition(", ")
    if pv_adresse:
        esc = lambda x: x.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        lines.append(
            f'<{t}:p {t}:style-name="PV_ZEILE1">- Prozessbevollmächtigte:'
            f'<{t}:tab/>{esc(pv_name)},</{t}:p>'
        )
        lines.append(P(f"{pv_adresse} -", "PV_EINZUG"))
    else:
        lines.append(P(f"- Prozessbevollmächtigte: {pv} -"))
    lines.append(P("gegen"))
    passiv = spec["passiv"]
    passiv_line = passiv["bezeichnung"].rstrip(",")
    if passiv.get("zusatz"):
        passiv_line += f', {passiv["zusatz"]}'
    if passiv.get("adresse"):
        passiv_line += f', {passiv["adresse"]}'
    lines.append(P(passiv_line + ","))
    lines.append(P(spec["passiv_rolle"].rstrip(",") + ",", "RUBRUM_RECHTS"))
    return "".join(lines)


def patch_odt(xml, ns, spec):
    """Rubrum-Block zwischen Verfahrens-Überschrift und 'wegen …' ersetzen."""
    xml = _ensure_styles(xml, ns)
    t = ns["text"]
    para_re = re.compile(r"<" + t + r":p [^>]*?(?<!/)>(.*?)</" + t + r":p>|<" + t + r":p [^>]*/>", re.DOTALL)
    heading_re = HEADING_RE
    paras = list(para_re.finditer(xml))
    start = ende = None
    for m in paras:
        text = _para_text(m.group(1) or "")
        if start is None and heading_re.match(text):
            start = m
            continue
        if start is not None and text.lower().startswith("wegen"):
            ende = m
            break
    if start is None or ende is None:
        raise SystemExit("Rubrum-Block nicht gefunden (Überschrift oder 'wegen …' fehlt)")
    rubrum = build_rubrum_paragraphs(spec, ns)
    xml = xml[: start.end()] + rubrum + xml[ende.start():]
    print("Rubrum ersetzt", file=sys.stderr)
    return xml


# ---------------------------------------------------------------------------
# spec-from-case: Rubrum-Spezifikation direkt aus den j-lawyer-Parteien bauen.
# Keine Handeingabe von Parteien — Quelle ist die Akte (jlawyer-cli parties).
# ---------------------------------------------------------------------------

JLAWYER_CLI = "/home/jay/.codex/skills/api/scripts/jlawyer-cli"

# Rollenpaare je Verfahrenstyp: (aktiv_m, aktiv_w, aktiv_pl, passiv)
TYP_ROLLEN = {
    "klage": ("Kläger", "Klägerin", "Kläger", "Beklagte"),
    "klage80v": ("Kläger und Antragsteller", "Klägerin und Antragstellerin",
                 "Kläger und Antragsteller", "Beklagte und Antragsgegnerin"),
    "eilantrag": ("Antragsteller", "Antragstellerin", "Antragsteller", "Antragsgegnerin"),
    "beschwerde": ("Beschwerdeführer", "Beschwerdeführerin", "Beschwerdeführer",
                   "Beschwerdegegnerin"),
    "widerspruch": ("Widerspruchsführer", "Widerspruchsführerin", "Widerspruchsführer",
                    "Widerspruchsgegnerin"),
}

BRD_BAMF_ZUSATZ = ("vertreten durch den Bundesminister des Inneren, dieser vertreten "
                   "durch den Präsidenten des Bundesamtes für Migration und Flüchtlinge "
                   "in 90343 Nürnberg")

# Kreisfreie Städte in NRW → Oberbürgermeister, alle übrigen → Bürgermeister.
KREISFREI_NRW = {
    "Bielefeld", "Bochum", "Bonn", "Bottrop", "Dortmund", "Duisburg",
    "Düsseldorf", "Essen", "Gelsenkirchen", "Hagen", "Hamm", "Herne", "Köln",
    "Krefeld", "Leverkusen", "Mönchengladbach", "Mülheim an der Ruhr",
    "Münster", "Oberhausen", "Remscheid", "Solingen", "Wuppertal", "Aachen",
}


def _partei_adresse(c):
    strasse = " ".join(x for x in (c.get("street", ""), c.get("streetNumber", "")) if x).strip()
    ort = " ".join(x for x in (c.get("zipCode", ""), c.get("city", "")) if x).strip()
    return ", ".join(x for x in (strasse, ort) if x)


def spec_from_case(case_ref, typ="klage", aktiv_rolle=None, passiv_rolle=None,
                   passiv_zusatz=None):
    import subprocess
    raw = subprocess.run([JLAWYER_CLI, "parties", case_ref, "--json"],
                         capture_output=True, text=True, check=True).stdout
    parties = json.loads(raw)
    mandanten = [p["contact"] for p in parties if p["involvementType"] == "Mandant"]
    gegner = [p["contact"] for p in parties if p["involvementType"] == "Gegner"]
    if not mandanten:
        raise SystemExit(f"Keine Mandanten-Partei in {case_ref}")
    if not gegner:
        raise SystemExit(f"Keine Gegner-Partei in {case_ref}")

    a_m, a_w, a_pl, p_std = TYP_ROLLEN[typ]
    if aktiv_rolle is None:
        if len(mandanten) > 1:
            aktiv_rolle = a_pl
        else:
            aktiv_rolle = a_w if mandanten[0].get("gender") == "FEMALE" else a_m

    g = gegner[0]
    g_name = g.get("company") or f'{g.get("firstName", "")} {g.get("name", "")}'.strip()
    if passiv_zusatz is None:
        if "Bundesamt für Migration" in g_name or "Bundesrepublik" in g_name:
            g_name = "die Bundesrepublik Deutschland"
            passiv_zusatz = BRD_BAMF_ZUSATZ
        elif g_name.startswith("Stadt "):
            ob = g_name[len("Stadt "):].strip() in KREISFREI_NRW
            g_name = f"die {g_name}"
            passiv_zusatz = ("vertreten durch den Oberbürgermeister" if ob
                             else "vertreten durch den Bürgermeister")
        elif g_name.startswith("Kreis "):
            g_name = f"der {g_name}"
            passiv_zusatz = "vertreten durch den Landrat"

    spec = {
        "aktiv_rolle": aktiv_rolle,
        "passiv_rolle": passiv_rolle or p_std,
        "aktiv": [
            {
                # Genitiv wie im Vorbild: mehrere Parteien "des Herrn X" /
                # "der Frau Y", Einzelpartei "des X" / "der Y" (Ahmadi 011/26).
                "bezeichnung": (
                    (("des Herrn " if len(mandanten) > 1 else "des ")
                     if m.get("gender") != "FEMALE"
                     else ("der Frau " if len(mandanten) > 1 else "der "))
                    + f'{m.get("firstName", "")} {m.get("name", "")}'.strip()
                ),
                "geb": m.get("birthDate", ""),
                "adresse": _partei_adresse(m),
            }
            for m in mandanten
        ],
        "passiv": {"bezeichnung": g_name},
    }
    if passiv_zusatz:
        spec["passiv"]["zusatz"] = passiv_zusatz
    adresse_g = _partei_adresse(g)
    if adresse_g and "Bundesrepublik" not in g_name:
        spec["passiv"]["adresse"] = adresse_g
    return spec


# ---------------------------------------------------------------------------
# check: Kanzlei-Formatierung deterministisch verifizieren (Lint).
# ET-basiert mit voller Stilauflösung (styles.xml + automatic styles + parents
# + Span-Ebene), unabhängig von Stilnamen — geprüft werden die EFFEKTIVEN
# Eigenschaften (fett/zentriert/rechtsbündig/unterstrichen/eingerückt).
# ---------------------------------------------------------------------------

NS_TEXT = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
NS_STYLE = "urn:oasis:names:tc:opendocument:xmlns:style:1.0"
NS_FO = "urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0"
NS_OFFICE = "urn:oasis:names:tc:opendocument:xmlns:office:1.0"


def _parse_xml(data):
    """XML sicher parsen: ODF enthält nie DOCTYPE/Entities — bei Fund abbrechen
    (Schutz vor XXE/billion-laughs in fremden Dateien, Tool bleibt stdlib-only)."""
    import xml.etree.ElementTree as ET
    head = data[:4096] if isinstance(data, bytes) else data[:4096].encode()
    if b"<!DOCTYPE" in head or b"<!ENTITY" in head:
        raise SystemExit("Verweigert: XML mit DOCTYPE/ENTITY (kein gültiges ODF)")
    return ET.fromstring(data)


def _load_styles(path):
    z = zipfile.ZipFile(path)
    styles = {}
    for part in ("styles.xml", "content.xml"):
        root = _parse_xml(z.read(part))
        for st in root.iter(f"{{{NS_STYLE}}}style"):
            styles[st.get(f"{{{NS_STYLE}}}name")] = st
    content = _parse_xml(z.read("content.xml"))
    body = content.find(f"{{{NS_OFFICE}}}body/{{{NS_OFFICE}}}text")
    return styles, body


def _props(styles, name, _seen=None):
    _seen = _seen or set()
    if not name or name in _seen or name not in styles:
        return {}
    _seen.add(name)
    el = styles[name]
    out = _props(styles, el.get(f"{{{NS_STYLE}}}parent-style-name", ""), _seen)
    for pp in el.findall(f"{{{NS_STYLE}}}paragraph-properties"):
        for k in ("text-align", "margin-left", "text-indent"):
            v = pp.get(f"{{{NS_FO}}}{k}")
            if v:
                out[k] = v
    for tp in el.findall(f"{{{NS_STYLE}}}text-properties"):
        w = tp.get(f"{{{NS_FO}}}font-weight")
        if w:
            out["weight"] = w
        u = tp.get(f"{{{NS_STYLE}}}text-underline-style")
        if u:
            out["underline"] = u
    return out


def _cm(v):
    """Längenangabe nach cm ('0.5in' → 1.27)."""
    m = re.match(r"(-?[\d.]+)(cm|in|mm|pt)", v or "")
    if not m:
        return 0.0
    n = float(m.group(1))
    return {"cm": n, "in": n * 2.54, "mm": n / 10, "pt": n * 0.03528}[m.group(2)]


def _effective(styles, p):
    """Effektive Absatz-Eigenschaften inkl. Span-Fett/-Unterstreichung über den ganzen Text."""
    pr = dict(_props(styles, p.get(f"{{{NS_TEXT}}}style-name", "")))
    spans = p.findall(f"{{{NS_TEXT}}}span")
    text = "".join(p.itertext()).strip()
    if spans and text:
        stext = "".join("".join(s.itertext()) for s in spans).strip()
        if stext == text:
            sp = [_props(styles, s.get(f"{{{NS_TEXT}}}style-name", "")) for s in spans]
            if sp and all(x.get("weight") == "bold" for x in sp):
                pr["weight"] = "bold"
            if sp and all(x.get("underline") not in (None, "none") for x in sp):
                pr["underline"] = "solid"
    bold = pr.get("weight") == "bold"
    underline = pr.get("underline") not in (None, "none")
    align = pr.get("text-align", "start")
    indent = _cm(pr.get("margin-left", "0cm"))
    # Führende Tabs zählen als Einzug (ältere Dokumente rücken damit ein,
    # ca. 1.25cm pro Tabstopp).
    if not (p.text or "").strip():
        for child in p:
            if child.tag == f"{{{NS_TEXT}}}tab":
                indent += 1.25
                if (child.tail or "").strip():
                    break
            else:
                break
    return text, bold, underline, align, indent


def check_odt(path, behoerde=False):
    """Regeln des Formatvorbilds prüfen. Rückgabe: Liste (regel, text, ist, soll)."""
    styles, body = _load_styles(path)
    paras = [c for c in body if c.tag == f"{{{NS_TEXT}}}p"]
    rows_all = [_effective(styles, p) for p in paras]
    rows = [r for r in rows_all if r[0]]
    fails = []

    def fail(regel, text, ist, soll):
        fails.append((regel, text[:70], ist, soll))

    # Leerzeilen-Grammatik (Vorbild Keienborg): nach 'wegen …', nach dem
    # Antrag-Lead-in und nach jedem Antragszonen-Element folgt eine Leerzeile.
    zone = False  # False | "erster" | "weiter" (wie die Hauptzustandsmaschine)
    for j, (text, *_rest) in enumerate(rows_all):
        if not text:
            continue
        braucht_leer = False
        grund = None
        if behoerde:
            # Jay-Ansage 14.07.2026: Fließtext ohne Pflicht-Leerzeilen
            # (Gerichts-Grammatik), nur Strukturgrenzen sind Pflicht.
            if GRUSS_RE.match(text):
                if j > 0 and (rows_all[j - 1][0] or "").strip():
                    fail("Leerzeile vor Grußformel", text,
                         "Absatz direkt davor", "eine Leerzeile")
            if re.match(r"^Sehr geehrte", text):
                braucht_leer, grund = True, "Leerzeile nach der Anrede"
        # Leerzeile vor jeder unterstrichenen Überschrift (Jay, 11.08.2026) —
        # eigenständiges if, nicht Teil der Zonen-Kette darunter: die
        # verbraucht BEGRUENDUNG_RE im ersten Zweig.
        if j > 0 and ist_unterstrichene_ueberschrift(text) \
                and (rows_all[j - 1][0] or "").strip():
            fail("Leerzeile vor unterstrichener Überschrift", text,
                 "Absatz direkt davor", "eine Leerzeile davor")
        if BEGRUENDUNG_RE.match(text) or GRUSS_RE.match(text):
            zone = False
        elif WEGEN_RE.match(text):
            braucht_leer, grund = True, "Leerzeile nach 'wegen …'"
        elif ist_leadin(text):
            zone = "erster"
            braucht_leer, grund = True, "Leerzeile nach dem Antrag-Lead-in"
        elif zone == "erster":
            braucht_leer, grund = True, "Leerzeile nach Antrag/hilfsweise"
            zone = "erster" if HILFSWEISE_RE.match(text) else "weiter"
        elif zone == "weiter":
            if HILFSWEISE_RE.match(text):
                braucht_leer, grund = True, "Leerzeile nach Antrag/hilfsweise"
                zone = "erster"
            elif NUMMERIERT_RE.match(text):
                braucht_leer, grund = True, "Leerzeile nach Antrag/hilfsweise"
            else:
                zone = False
        naechster_leer = j + 1 < len(rows_all) and not rows_all[j + 1][0]
        if braucht_leer and not naechster_leer and j + 1 < len(rows_all):
            fail(grund, text, "direkt gefolgt von Text", "eine Leerzeile dazwischen")

    # Leerzeichen am Absatzende (Platzhalter-Artefakt, z.B. 'Az.: … ') — der
    # gestrippte rows-Text kann das nicht sehen, daher Roh-Durchlauf.
    for p in paras:
        raw = "".join(p.itertext())
        if raw.strip() and raw != raw.rstrip():
            fail("Leerzeichen-Artefakte", raw.strip(), "Leerzeichen am Absatzende",
                 "Platzhalter-Reste bereinigen (rubrum-cli format)")

    pv_folgezeile = False
    anlagen_referenziert = any(ANLAGE_REF_RE.search(t) for t, *_ in rows
                               if not ANLAGE_RE.match(t))
    anreden = [t for t, *_ in rows if re.match(r"^Sehr geehrte", t)]
    if len(anreden) > 1:
        fail("Anrede nur einmal", anreden[1], f"{len(anreden)}× 'Sehr geehrte …'",
             "einmal (Vorlage enthält sie oft schon — Body darf sie nicht wiederholen)")
    in_antraege = False
    aufzaehlung_offen = False
    rubrum_state = None
    for i, (text, bold, ul, align, indent) in enumerate(rows):
        if rubrum_state == "partei":
            if not bold:
                fail("Parteienzeile fett", text, "nicht fett", "fett")
            rubrum_state = "az"
        elif rubrum_state == "az":
            if AZ_RE.match(text) and not ul:
                fail("Az. unterstrichen", text, "nicht unterstrichen", "unterstrichen")
            rubrum_state = None
        elif ROLLEN_RE.match(text):
            if align not in ("right", "end"):
                fail("Rollenzeile rechtsbündig", text, align, "right")
        elif not in_antraege and HEADING_RE.match(text):
            if not (bold and align == "center"):
                fail("Verfahrensüberschrift fett+zentriert", text,
                     f"fett={bold}, align={align}", "fett, center")
        elif BEGRUENDUNG_RE.match(text):
            if not (bold and ul):
                fail("'Begründung:' fett+unterstrichen", text,
                     f"fett={bold}, unterstrichen={ul}", "fett, unterstrichen")
        elif behoerde and ZUR_BEGRUENDUNG_RE.match(text):
            if align != "center":
                fail("'Zur Begründung:' zentriert", text,
                     f"align={align}", "center (nicht fett)")
        elif FRIST_DATUM_RE.match(text):
            if not (bold and align == "center"):
                fail("Fristsetzung fett+zentriert", text,
                     f"fett={bold}, align={align}", "fett, center (eigener Absatz)")
        elif BEWEIS_RE.match(text):
            if not bold or indent < 0.6:
                fail("Beweis/Anlagen-Referenz fett+eingerückt", text,
                     f"fett={bold}, Einzug={indent:.2f}cm", "fett, Einzug ≥ 1cm (Vorbild: 1.27cm)")
        elif ABSCHNITT_RE.match(text) and len(text) < 60:
            if not (align == "center" and ul):
                fail("Abschnittsüberschrift zentriert+unterstrichen", text,
                     f"align={align}, unterstrichen={ul}", "center, unterstrichen")
        elif in_antraege and GRUSS_RE.match(text):
            in_antraege = False
        elif in_antraege and HILFSWEISE_RE.match(text):
            if bold:
                fail("'hilfsweise,' nicht fett", text, "fett", "nicht fett")
            in_antraege = "erster"
        elif in_antraege and NUMMERIERT_RE.match(text):
            if not bold or indent < 0.6:
                fail("Antrag fett+eingerückt", text,
                     f"fett={bold}, Einzug={indent:.2f}cm", "fett, Einzug ≥ 1cm")
            in_antraege = "weiter"
        elif in_antraege == "erster":
            if AKTENEINSICHT_RE.match(text):
                if not (bold and align == "center"):
                    fail("Akteneinsicht-Antrag fett+zentriert", text,
                         f"fett={bold}, align={align}", "fett, center")
            elif not bold or indent < 0.6:
                fail("Antrag fett+eingerückt", text,
                     f"fett={bold}, Einzug={indent:.2f}cm, align={align}",
                     "fett, Einzug ≥ 1cm (Akteneinsicht-Anträge stattdessen zentriert)")
            in_antraege = "weiter"
        elif in_antraege == "weiter":
            in_antraege = False
        elif aufzaehlung_offen and NUMMERIERT_RE.match(text):
            if indent < 0.6:
                fail("Aufzählungspunkt eingerückt", text,
                     f"Einzug={indent:.2f}cm", "Einzug ≥ 1cm (Vorbild: 1.27cm), nicht fett")
        elif PV_RE.match(text):
            if text.rstrip().endswith("-"):
                fail("Prozessbevollmächtigte zweizeilig", text,
                     "einzeilig", "Zeile 1 Name, Zeile 2 Adresse eingerückt (Vorbild 011/26)")
            else:
                pv_folgezeile = True
        elif pv_folgezeile:
            pv_folgezeile = False
            if indent < 4.0:
                fail("PV-Adresszeile eingerückt", text,
                     f"Einzug={indent:.2f}cm", "Einzug ≥ 4cm (Vorbild: 6.35cm)")
        elif ANLAGE_RE.match(text):
            if not ul:
                fail("'Anlage(n)' unterstrichen", text, "nicht unterstrichen", "unterstrichen")
            if not anlagen_referenziert:
                fail("Anlagenzeile ohne Anlagen", text,
                     "'Anlage(n)' unter der Signatur, aber keine Anlage im Text referenziert",
                     "Zeile entfernen (rubrum-cli format)")
        if AZ_LEER_RE.match(text):
            fail("Leere Az.-Zeile", text, "Az.: ohne Inhalt",
                 "Az. eintragen oder Zeile entfernen (Vorbild 158/25)")
        if (re.search(r"\S ,", text) or "  " in text) and not re.match(r"[Vv]gl\.", text):
            fail("Leerzeichen-Artefakte", text, "' ,' oder Doppel-Leerzeichen",
                 "Platzhalter-Reste bereinigen (rubrum-cli format)")
        if ";" in text and not re.match(r"[Vv]gl\.", text):
            # 'Vgl. …'-Fundstellen trennen Entscheidungen konventionsgemäß mit ';'
            fail("Kein Semikolon", text, "enthält ';'", "eigenständige Sätze")
        if VERFAHREN_LEADIN_RE.match(text):
            rubrum_state = "partei"
        # Nur ECHTE Lead-ins armieren: sie enden auf ',' oder ':'
        # (inline erledigte Anträge wie 'Zugleich beantragen wir Einsicht … .' nicht).
        if ist_leadin(text):
            in_antraege = "erster"
            aufzaehlung_offen = False
        elif _kein_aufzaehlung_leadin(text):
            aufzaehlung_offen = False
        elif AUFZAEHLUNG_LEADIN_RE.search(text):
            aufzaehlung_offen = True
        elif not NUMMERIERT_RE.match(text):
            aufzaehlung_offen = False
        if BEGRUENDUNG_RE.match(text) or ZUR_BEGRUENDUNG_RE.match(text):
            in_antraege = False
    return fails


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("format", help="Kanzlei-Formatierung erzwingen (in-place ohne -o)")
    f.add_argument("odt")
    f.add_argument("-o", "--out")
    f.add_argument("--behoerde", action="store_true",
                   help="Behördenschreiben: Antrag nach Lead-in fett+zentriert")

    c = sub.add_parser("check", help="Kanzlei-Formatierung verifizieren (Exit 1 bei Verstößen)")
    c.add_argument("odt", nargs="+")
    c.add_argument("--behoerde", action="store_true",
                   help="Behördenschreiben: Antrag muss fett+zentriert sein")

    p = sub.add_parser("patch", help="Rubrum aus JSON-Spez einsetzen + formatieren")
    p.add_argument("odt")
    p.add_argument("--spec", required=True, help="JSON-Datei (siehe SKILL.md) oder - für stdin")
    p.add_argument("-o", "--out")

    b = sub.add_parser("block", help="Rubrum als Klartext ausgeben (Review)")
    b.add_argument("--spec", required=True)

    sf = sub.add_parser("spec-from-case",
                        help="Rubrum-Spez aus den j-lawyer-Parteien der Akte bauen (JSON auf stdout)")
    sf.add_argument("case_ref")
    sf.add_argument("--typ", choices=sorted(TYP_ROLLEN), default="klage")
    sf.add_argument("--aktiv-rolle", help="Rolle der Aktivpartei überschreiben")
    sf.add_argument("--passiv-rolle", help="Rolle der Passivpartei überschreiben")
    sf.add_argument("--passiv-zusatz", help="Vertretungszusatz der Passivpartei überschreiben")

    args = ap.parse_args()

    if args.cmd == "spec-from-case":
        spec = spec_from_case(args.case_ref, typ=args.typ, aktiv_rolle=args.aktiv_rolle,
                              passiv_rolle=args.passiv_rolle, passiv_zusatz=args.passiv_zusatz)
        print(json.dumps(spec, ensure_ascii=False, indent=2))
        return

    if args.cmd == "check":
        total = 0
        for path in args.odt:
            fails = check_odt(path, behoerde=args.behoerde)
            total += len(fails)
            if len(args.odt) > 1:
                print(f"--- {path}")
            if not fails:
                print("OK: alle Konventionen eingehalten")
            for regel, text, ist, soll in fails:
                print(f"VERSTOSS [{regel}] „{text}“ — ist: {ist}, soll: {soll}")
        sys.exit(1 if total else 0)

    if args.cmd == "block":
        spec = json.load(sys.stdin if args.spec == "-" else open(args.spec, encoding="utf-8"))
        fake_ns = {"text": "text", "style": "style", "fo": "fo", "auto": "office"}
        frag = build_rubrum_paragraphs(spec, fake_ns)
        for m in re.finditer(r"<text:p [^>]*?>(.*?)</text:p>", frag, re.DOTALL):
            style = re.search(r'style-name="([^"]+)"', m.group(0)).group(1)
            align = " [rechtsbündig]" if style == "RUBRUM_RECHTS" else ""
            print(_para_text(m.group(1)) + align)
        return

    src, xml = _read(args.odt)
    ns = _prefixes(xml)
    if args.cmd == "patch":
        spec = json.load(sys.stdin if args.spec == "-" else open(args.spec, encoding="utf-8"))
        xml = patch_odt(xml, ns, spec)
        xml = format_odt(xml, ns)
    else:
        xml = format_odt(xml, ns, behoerde=args.behoerde)
    out_path = args.out or args.odt
    if out_path == args.odt:
        tmp = args.odt + ".tmp"
        _write(src, xml, tmp)
        src.close()
        shutil.move(tmp, args.odt)
    else:
        _write(src, xml, out_path)
        src.close()
    print(f"geschrieben: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
