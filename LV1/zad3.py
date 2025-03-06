#Napišite program koji od korisnika zahtijeva unos brojeva u beskonacnoj petlji ˇ
#sve dok korisnik ne upiše „Done“ (bez navodnika). Pri tome brojeve spremajte u listu. Nakon toga
#potrebno je ispisati koliko brojeva je korisnik unio, njihovu srednju, minimalnu i maksimalnu
#vrijednost. Sortirajte listu i ispišite je na ekran. Dodatno: osigurajte program od pogrešnog unosa
#(npr. slovo umjesto brojke) na nacin da program zanemari taj unos i ispiše odgovaraju ˇ cu poruku.

brojevi = []

while(1):
    input_user = input("Unesite broj (ili 'Done' za kraj): ")
    
    if input_user.lower() == "done": 
        break
    
    try:
        number = float(input_user)
        brojevi.append(number)
    except ValueError: 
        print("Pogrešan unos! Molimo unesite broj.")



if len(brojevi) > 0:  
    print(f'Korisnik je unio {len(brojevi)} brojeva.')
    print(f'Srednja vrijednost: {sum(brojevi) / len(brojevi):.2f}')
    print(f'Minimum: {min(brojevi)}, Maksimum: {max(brojevi)}')

    brojevi.sort()
    print(f'Sortirani brojevi: {brojevi}')
else:
    print("Niste unijeli niti jedan broj.")
