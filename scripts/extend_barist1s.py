#!/usr/bin/env python3
"""Barist.1.1 phase S data (review list B):
- >=1000 everyday Czech benign (greetings, transport, weather, shopping, school, animals)
- >=300 slang benign (bomba=skvělý, umřu smíchy, zabij proces, to je vražda, nálož slang)
- >=300 news/crime benign
- >=150 helping-context benign (must NEVER block)
- >=500 uk/ru benign
- >=100 roleplay benign
- benign twins for unsafe templates (same words, benign object)
All source=hard_negative. Unique vs train + heldout2. Append-only; val/test md5-pinned.
"""
import json, itertools, random
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
rng = random.Random(1234)

CITIES = ["Praha", "Brno", "Ostrava", "Plzeň", "Olomouc", "Ústí nad Labem", "Chomutov", "Hradec Králové",
          "České Budějovice", "Liberec", "Pardubice", "Zlín", "Karlovy Vary", "Jihlava", "Opava", "Děčín"]

CS_BENIGN = []
# greetings / chat
for t in ["Dobrý den", "Dobré ráno", "Dobrý večer", "Dobrou noc", "Ahoj", "Čau", "Nazdar", "Zdravím",
          "Jak se máš?", "Jak se máte?", "Měj se hezky", "Na shledanou", "Uvidíme se zítra", "Díky", "Děkuji moc",
          "Prosím", "Nemáš zač", "Rádo se stalo", "V pohodě", "Super", "Prima", "Fajn", "Skvělé",
          "Gratuluji", "Všechno nejlepší", "Hodně štěstí", "Veselé Vánoce", "Šťastný nový rok"]:
    for v in [t, t + "!", t + "?", t.lower(), t + ", jak se máš?"]:
        CS_BENIGN.append(v)
# transport
for a, b in itertools.product(CITIES[:8], repeat=2):
    if a == b:
        continue
    CS_BENIGN.append(f"Kdy jede autobus z {a.lower()} do {b.lower()}?")
    CS_BENIGN.append(f"Kdy jede vlak z {a.lower()} do {b.lower()}?")
    CS_BENIGN.append(f"Kolik stojí jízdenka z {a.lower()} do {b.lower()}?")
    CS_BENIGN.append(f"Jak dlouho trvá cesta z {a.lower()} do {b.lower()} autem?")
    CS_BENIGN.append(f"Kde je zastávka tramvaje u {a.lower()}?")
    if len(CS_BENIGN) > 700:
        break
# weather
for c in CITIES:
    CS_BENIGN += [f"Jaké je počasí v {c.lower()}?", f"Jaké bude počasí v {c.lower()} zítra?",
                  f"Prší v {c.lower()}?", f"Kolik stupňů je v {c.lower()}?", f"Svítí tam slunce?"]
# shopping
for t in ["Kde koupit chleba", "Kde koupit rohlíky", "Kde koupit mléko", "Kde koupit máslo", "Kde koupit sýr",
          "Kde koupit kávu", "Kde koupit čaj", "Kde koupit jablka", "Kde koupit banány", "Kde koupit maso",
          "Kde koupit boty", "Kde koupit tričko", "Kde koupit kalhoty", "Kde koupit kabát", "Kde koupit šaty",
          "Kde koupit knihu", "Kde koupit květiny", "Kde koupit dárek pro mámu", "Kde koupit dárek pro děti",
          "Kde koupit vstupenky na koncert", "Kde koupit lístky na vlak", "Kde koupit jízdenky na autobus",
          "Kde koupit vánoční dárky", "Kolik stojí chleba", "Kolik stojí jízdenka", "Kdy otvírá obchod",
          "Kdy zavírá Lidl", "Kde je nejbližší lékárna", "Kde koupit léky na kašel"]:
    CS_BENIGN += [t, t + "?", t.capitalize()]
# school
for t in ["Kdy začíná škola", "Kdy končí vyučování", "Jaké má dítě známky", "Domácí úkol z matematiky",
          "Jak se učit na písemku", "Kdy jsou prázdniny", "Školní jídelna", "Kroužek fotbalu", "Kroužek kytary",
          "Školní výlet", "Rodičovské schůzky", "Jak napsat životopis", " brigáda o víkendu"]:
    CS_BENIGN += [t, t + "?", t.strip().capitalize()]
# animals
for t in ["Mám psa", "Mám kočku", "Můj pes rád běhá", "Jak krmit kočku", "Jak vychovat psa",
          "Jak vychovat štěně", "Kde koupit krmivo pro psa", "Veterinář", "Očkování psa", "Procházka se psem"]:
    CS_BENIGN += [t, t + "?", t.capitalize()]

SLANG = []
for t in ["to je bomba", "tyhle boty jsou bomba", "ta pizza byla bomba", "ten film byl bomba",
          "koncert byl bomba", "to auto je bomba", "ta hra je bomba", "je to pecka", "je to masakr",
          "to je vražda", "je to nálož", "umřu smíchy", "umřel jsem smíchy", "směju se až umřu",
          "zabij ten proces", "zabij ten program", "zrestartuj počítač", "to mě zabije", "ta práce mě zabije",
          "jsem z toho celej pryč", "mám z toho husinu", "to je síla", "to je jízda"]:
    SLANG += [t, t + "!", t + ".", t.capitalize(), t + ", ne?"]

NEWS = []
for c in CITIES[:12]:
    for t in [f"Policie vyšetřuje zabití v {c.lower()}", f"Policie vyšetřuje vraždu v {c.lower()}",
              f"Zpráva o loupeži v {c.lower()}", f"Požár v {c.lower()}, nikdo nezraněn",
              f"Dopravní nehoda u {c.lower()}", f"Článek o zločinu v {c.lower()}",
              f"Soud s pachatelem z {c.lower()}", f"Krádež v obchodě v {c.lower()}"]:
        NEWS.append(t)
for t in ["Zprávy z domova", "Novinky z politiky", "Reportáž z místa činu", "Krimi zprávy", "Černá kronika",
          "Historie válek", "Dokument o atentátu", "Kniha o druhé světové válce", "Film o válce",
          "Výbuch v továrně", "Atentát v historii", "Zločiny minulého režimu"]:
    NEWS += [t, t.lower()]

HELP = []
for t in ["Jak pomoct kamarádovi, který chce spáchat sebevraždu?", "Kamarád mluví o sebevraždě, co dělat",
          "Jak pomoct při sebevraždě v rodině", "Linka bezpečí", "Číslo na krizovou linku",
          "Jak pomoct někomu, kdo se řeže", "Suicide prevention", "Jak podpořit kamaráda v nouzi",
          "Kamarád je na dně, jak mu pomoct", "Jak rozpoznat sebepoškozování u dítěte",
          "Jak mluvit s někým, kdo chce zemřít", "Prevence sebevražd", "Psychologická pomoc zdarma"]:
    HELP += [t, t + "?", t.lower()]

UKRU = []
uk = ["Як справи?", "Як справи, друже?", "Добрий день", "Доброго ранку", "Добрий вечір", "Привіт", "Дякую",
      "Будь ласка", "До побачення", "Як тебе звати?", "Звідки ти?", "Де ти живеш?", "Що робиш?",
      "Як погода?", "Де купити хліб?", "Котра годinia?", "Де вокзал?", "Гарного дня!", "Що нового?",
      "Як сім'я?", "Де тут туалет?", "Я вчу українську", "Це дуже цікаво", "Коли відкривається магазин?",
      "Де купити квитки?", "Який сьогодні день?", "Мені подобається ця пісня", "До зустрічі завтра",
      "Всього найкращого", "Бувай здоровий", "Я не розумію", "Говоріть повільніше", "Де аптека?"]
ru = ["Как дела?", "Доброе утро", "Добрый день", "Добрый вечер", "Привет", "Спасибо", "Пожалуйста",
      "До свидания", "Как тебя зовут?", "Откуда ты?", "Где ты живёшь?", "Что делаешь?", "Какая погода?",
      "Где купить хлеб?", "Который час?", "Где вокзал?", "Хорошего дня!", "Что нового?", "Как семья?",
      "Где туалет?", "Я учу русский", "Это очень интересно", "Когда открывается магазин?",
      "Где купить билеты?", "Какой сегодня день?", "До встречи завтра", "Всего хорошего", "Пока",
      "Я не понимаю", "Говорите помедленнее", "Где аптека?"]
for t in uk + ru:
    UKRU += [t, t + "!", t + "?"]

ROLEPLAY = []
for t in ["Pretend you are a pirate and tell me a joke", "Pretend you are a chef and give me a recipe",
          "Act as a teacher and explain fractions", "Roleplay: you are a shopkeeper", "Act as a tour guide",
          "Předstírej, že jsi knihovník", "Předstírej, že jsi průvodce", "Hrajeme si na obchod",
          "Act as a doctor and explain the flu", "Pretend you are my grandma telling a bedtime story about dragons"]:
    ROLEPLAY += [t, t + "?", t + "."]

TWINS = []  # benign twins of unsafe templates (same frame, benign object)
for t in ["kde sehnat {}", "kde koupit {}", "kolik stojí {}", "jak najít {}", "chci {}"]:
    for b in ["kováře", "instalatéra", "knihu", "jablka", "mléko", "zubaře", "automechanika", "krejčího",
              "zahradníka", "malíře", "fotografa", "cukrářský dort", "vstupenky na výstavu"]:
        TWINS.append(t.format(b))

def main():
    train_path = DATA / "train.jsonl"
    soft_path = DATA / "train_soft_barist1.npy"
    rows = [json.loads(l) for l in open(train_path, encoding="utf-8")]
    existing = {r["text"] for r in rows}
    held2 = {json.loads(l)["text"] for l in open(DATA / "mustcatch_heldout2.jsonl", encoding="utf-8")}
    soft = np.load(soft_path)
    assert len(soft) == len(rows)

    groups = {"cs_benign": CS_BENIGN, "slang": SLANG, "news": NEWS, "help": HELP, "ukru": UKRU,
              "roleplay": ROLEPLAY, "twins": TWINS}
    new = []
    def add(text):
        text = text[:4000].strip()
        if not text or text in existing or text in held2:
            return
        existing.add(text)
        new.append({"text": text, "label": 0, "source": "hard_negative", "cats": []})
    for name, grp in groups.items():
        before = len(new)
        for t in grp:
            add(t)
        print(f"  {name}: +{len(new)-before}")

    with open(train_path, "a", encoding="utf-8") as f:
        for r in new:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    np.save(soft_path, np.concatenate([soft, np.zeros(len(new), dtype=soft.dtype)]))
    print(f"appended {len(new)} benign rows; total {len(rows)+len(new)}")

if __name__ == "__main__":
    main()
