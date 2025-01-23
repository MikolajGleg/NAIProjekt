# NAI Projekt - Streszczanie Tekstów dla Biblioteki
Modele użyte:  
https://huggingface.co/facebook/bart-large-cnn  
https://huggingface.co/Falconsai/text_summarization  
https://huggingface.co/pszemraj/led-large-book-summary  
https://huggingface.co/gsarti/it5-base-news-summarization  

# Instrukcje Użycia  
W folderze którym chcemy otworzyć projekt:  
1)Otwieramy cmd  
2)Wpisujemy:  
`git clone https://github.com/MikolajGleg/NAIProjekt.git`   
3)Zainstaluj python przez linka poniżej (troche starsza wersja aby pytorch był kompatybilny)  
[installer python ](https://www.python.org/downloads/release/python-3100/  )  

4)Otwieramy ponownie terminal w folderze projektu i wpisujemy po kolei komendy:  
`py get-pip.py`  
`py -m pip install torch`  
`py -m pip install transformers`   
`py -m pip install rouge_score`   

5)Końcowym krokiem jest wpisanie komendy:  
**`py ./main.py`**  

