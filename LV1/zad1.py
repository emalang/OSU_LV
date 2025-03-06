#1 Napišite program koji od korisnika zahtijeva unos radnih sati te koliko je placen ´
#po radnom satu. Koristite ugradenu Python metodu ¯ input(). Nakon toga izracunajte koliko ˇ
#je korisnik zaradio i ispišite na ekran. Na kraju prepravite rješenje na nacin da ukupni iznos ˇ
#izracunavate u zasebnoj funkciji naziva ˇ total_euro.

working_hours=int(input("Radni sati:"))
payment_per_hour=float(input("eur/h:"))

def total_euro(working_hours, payment_per_hour):
    return working_hours*payment_per_hour

print(f'Ukupan iznos: {total_euro(working_hours,payment_per_hour)}')