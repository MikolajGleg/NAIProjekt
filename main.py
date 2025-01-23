from transformers import pipeline
from datasets import load_dataset
from rouge_score import rouge_scorer


#Stwórz Modele do streszczenia z hugging facea
#MODEL 1
summarizerModel1 = pipeline("summarization", model="facebook/bart-large-cnn")
#MODEL 2
summarizerModel2 = pipeline("summarization", model="Falconsai/text_summarization")
#MODEL 3
summarizerModel3 = pipeline("summarization", model="pszemraj/led-large-book-summary")
#MODEL 4
summarizerModel4 = pipeline("summarization", model='it5/it5-base-news-summarization')


#Porównaj modele używając ROUGE-1 czyli ile się pokrywa słów w faktycznym stresczeniu
#z generowanymi streszczeniami 

#Stwórz ROUGE-1 score
scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)


#Wczytujemy nasz dataset CNN/Daily Mail artykułów i ich streszczeń
dataset = load_dataset("cnn_dailymail", "3.0.0")

#Średnia wszystkich metryk inicjalizacja
total_precision = [0.0, 0.0, 0.0, 0.0]
total_recall = [0.0, 0.0, 0.0, 0.0]
total_f1 = [0.0, 0.0, 0.0, 0.0]

#Wyciągnij pierwsze 5 artykułów oraz ich dane streszczenia (reference_summary)
for i, example in enumerate(dataset['train']):
    if i >= 20:
        break
    document = example['article']
    reference_summary = example['highlights']

    
    #Skracanie dokumentu (jeśli jest za długi modele go nie sparsują)
    max_doc_length = 512
    if len(document.split()) > max_doc_length:
        document = ' '.join(document.split()[:max_doc_length])


    #Generacja streszczeń na podstawie naszych modelów
    summary1 = summarizerModel1(document, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
    summary2 = summarizerModel2(document, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
    summary3 = summarizerModel3(document, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
    summary4 = summarizerModel4(document, max_length=130, min_length=30, do_sample=False)[0]['summary_text']


    #Drukuj artykuł, dane streszczenie oraz wygenerowane streszczenia
    print(f"Document {i+1}:\n{document}\n")
    print(f"Reference Summary {i+1}:\n{reference_summary}\n")
    print(f"Model 1 Summary:\n{summary1}\n")
    print(f"Model 2 Summary:\n{summary2}\n")
    print(f"Model 3 Summary:\n{summary3}\n")
    print(f"Model 4 Summary:\n{summary4}\n")
    print("-" * 50)


    #Kalkulacja ROUGE-1 scorów
    scores1 = scorer.score(reference_summary, summary1)
    scores2 = scorer.score(reference_summary, summary2)
    scores3 = scorer.score(reference_summary, summary3)
    scores4 = scorer.score(reference_summary, summary4)

    #Zbierz na średnią wszystkich metryk
    total_precision[0] += scores1['rouge1'].precision
    total_precision[1] += scores2['rouge1'].precision
    total_precision[2] += scores3['rouge1'].precision
    total_precision[3] += scores4['rouge1'].precision

    total_recall[0] += scores1['rouge1'].recall
    total_recall[1] += scores2['rouge1'].recall
    total_recall[2] += scores3['rouge1'].recall
    total_recall[3] += scores4['rouge1'].recall

    total_f1[0] += scores1['rouge1'].fmeasure
    total_f1[1] += scores2['rouge1'].fmeasure
    total_f1[2] += scores3['rouge1'].fmeasure
    total_f1[3] += scores4['rouge1'].fmeasure

    #Drukuj ROUGE-1 scores dla wszystkich modeli
    print(f"Document {i+1} ROUGE-1 Scores:")
    print("Model 1 - Precision:", scores1['rouge1'].precision, "Recall:", scores1['rouge1'].recall, "F1 Score:", scores1['rouge1'].fmeasure)
    print("Model 2 - Precision:", scores2['rouge1'].precision, "Recall:", scores2['rouge1'].recall, "F1 Score:", scores2['rouge1'].fmeasure)
    print("Model 3 - Precision:", scores3['rouge1'].precision, "Recall:", scores3['rouge1'].recall, "F1 Score:", scores3['rouge1'].fmeasure)
    print("Model 4 - Precision:", scores4['rouge1'].precision, "Recall:", scores4['rouge1'].recall, "F1 Score:", scores4['rouge1'].fmeasure)
    print("-" * 50)

    #Interpretacja metryk ROUGE
    #--------------------------
    #Precyzja (Precision): Jak wiele z tekstu wygenerowanego jest istotne lub poprawne. Wysoka precyzja wskazuje, że większość zawartości w tekście kandydującym jest również zawarta w tekście referencyjnym, co oznacza, że model nie wprowadził wiele nieistotnych informacji.
    #Przykład: Jeśli streszczenie kandydujące zawiera dziesięć słów i 7 z nich jest również w streszczeniu referencyjnym, precyzja wynosi 0.7 (czyli 70%).
    #
    #Pamięć (Recall): Jak wiele z tekstu referencyjnego zostało uchwycone w tekście kandydującym. Wysoka pamięć wskazuje, że tekst kandydujący zawiera najważniejsze treści z tekstu referencyjnego, co jest kluczowe w zadaniach takich jak streszczenie, gdzie pominięcie kluczowych informacji jest problematyczne.
    #Przykład: Jeśli streszczenie referencyjne zawiera 12 słów, a 7 z tych słów jest również w streszczeniu kandydującym, pamięć wynosi 0.58 (czyli 58%).
    #
    #F1-Score (Wskaźnik F1): Wynik F1 balansuje precyzję i pamięć, dostarczając miary, która uwzględnia fałszywe alarmy (nieistotne treści zawarte) i fałszywe negatywy (istotne treści pominięte). Jest to szczególnie przydatne, gdy chcemy zapewnić zarówno wysoką trafność, jak i wszechstronność w wygenerowanym tekście.
    #Przykład: Korzystając z precyzji 70% i pamięci 58% z powyższych przykładów, wskaźnik F1 wynosiłby około 0.64 (czyli 64%).


# Podaj Średnie metryk
num_documents = 20
average_precision = [x / num_documents for x in total_precision]
average_recall = [x / num_documents for x in total_recall]
average_f1 = [x / num_documents for x in total_f1]

print("Average ROUGE-1 Scores for all models:")
print(f"Model 1 - Average Precision: {average_precision[0]}, Average Recall: {average_recall[0]}, Average F1 Score: {average_f1[0]}")
print(f"Model 2 - Average Precision: {average_precision[1]}, Average Recall: {average_recall[1]}, Average F1 Score: {average_f1[1]}")
print(f"Model 3 - Average Precision: {average_precision[2]}, Average Recall: {average_recall[2]}, Average F1 Score: {average_f1[2]}")
print(f"Model 4 - Average Precision: {average_precision[3]}, Average Recall: {average_recall[3]}, Average F1 Score: {average_f1[3]}")
