import Levenshtein
from random import choice
from random import randint

def generate_random_email():
    # random file number
    part1 = randint(100, 999)
    part2 = randint(1000, 999999)

    day = randint(1,31)
    month = randint(1,12)
    year = randint(1995, 2025)

    date = choice([True, False])


    # File number in the format "123/4567"
    file_number1 = f"{part1}/{part2}"
    file_number2 = f"Zeichen: {file_number1}"

    # Email template
    sender = "Von: Matthias Gander <matthias.gander@outlook.com>"
    email = "matthias.gander@outlook.com"
    receiver = "An: Max Müller <max.mueller@gmail.com>"
    subject = "Betreff: Mahnung XY"
    date1 = f"Datum: {day}.{month}.{year}" 
    date2 = f"Date: {day}-{month}-{year}"
    date3 = f"{year}/{month}/{day}"
    date4 = f"{day}.{month}"
    phone = "Tel. 0664/0986-3241"

    text = f"""
    {sender}
    {choice([receiver, ""])}
    Sprechzeiten: Montag - Freitag {randint(0,23)}:{randint(0,59)} - {randint(00,23)}:{randint(00,59)} Uhr
    Hallo,
    Im Anhang noch die Dateien.
    Absender:
    {sender}
    Achgasse 98, Dornbirn 6850
    {choice([phone, ""])}
    {email}

    {choice([date1, date2, date3])}

    {choice([receiver, ""])}

    {choice([file_number1, file_number2])}

    Sehr geehrte/r Herr/Frau Müller,

    trotz unserer vorherigen Korrespondenz und der freundlichen Erinnerung am {date4} haben wir bis zum heutigen Tage keinen Zahlungseingang Ihrer offenen Forderung feststellen können. Es handelt sich hierbei um die Rechnung Nr. {randint(100,9999)} vom {randint(1,31)}. über den Betrag von {randint(20,9999)}.

    Wir möchten Sie dringend darauf hinweisen, dass es sich hierbei um Ihre letzte Mahnung handelt, bevor wir rechtliche Schritte einleiten müssen. Um weitere Kosten und Unannehmlichkeiten zu vermeiden, bitten wir Sie, den ausstehenden Betrag bis spätestens nächster Woche auf das unten angegebene Konto zu überweisen:

    Bankverbindung:
    Kontoinhaber: Matthias Gander
    IBAN: AT{randint(123456789102345678, 999999999999999999)}
    BIC: PIAGAT2SXXX
    Verwendungszweck: Rechnungsnummer

    Sollten Sie den Betrag bereits überwiesen haben oder wenn es Unklarheiten gibt, bitten wir um umgehende Kontaktaufnahme unter {phone} oder per E-Mail an {email}.

    Bei Nichtbeachtung dieser Mahnung werden wir ohne weitere Ankündigung rechtliche Schritte einleiten. Hierdurch entstehen Ihnen zusätzliche Kosten, die wir Ihnen in Rechnung stellen werden.

    Wir hoffen auf eine zeitnahe Klärung dieser Angelegenheit und stehen für Rückfragen zur Verfügung.

    Mit freundlichen Grüßen,

    Unterschrift

    Mit freundlichen Grüßen,
    Matthias 
    """
    return text, file_number1


def accuracy(output, filenumber):
    true = 0
    for i in range(len(output)):
        if output[i] == filenumber[i]:
            true += 1
    accuracy = true/len(output)
    return accuracy, true, len(output)
        
def similarity(output, actual):
    true = 0
    for i in range(len(output)):
        distance = Levenshtein.distance(output[i], actual[i])
        print(distance)
        if distance <= 3:
            true += 1
    similarity_acc = true/len(output)
    return similarity_acc, true, len(output)
