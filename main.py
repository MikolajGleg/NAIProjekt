from transformers import pipeline
from datasets import load_dataset
from rouge_score import rouge_scorer

TEXT = """ The Battle of Alesia or siege of Alesia (September 52 BC) was the climactic military engagement of the Gallic Wars, fought around the Gallic oppidum (fortified settlement) of Alesia in modern France, a major centre of the Mandubii tribe. It was fought by the Roman army of Julius Caesar against a confederation of Gallic tribes united under the leadership of Vercingetorix of the Arverni. It was the last major engagement between Gauls and Romans, and is considered one of Caesar's greatest military achievements and a classic example of siege warfare and investment; the Roman army built dual lines of fortifications—an inner wall to keep the besieged Gauls in, and an outer wall to keep the Gallic relief force out. The Battle of Alesia marked the end of Gallic independence in the modern day territory of France and Belgium.

The battle site was probably atop Mont Auxois, above modern Alise-Sainte-Reine in France, but this location, some have argued, does not fit Caesar's description of the battle. A number of alternatives have been proposed over time, among which only Chaux-des-Crotenay (in Jura in modern France) remains a challenger today.[10]

The event is described by Caesar himself in his Commentarii de Bello Gallico as well as several later ancient authors (namely Plutarch and Cassius Dio). After the Roman victory, Gaul (very roughly modern France) was subdued, although Gallic territories north of Gallia Narbonensis would not become a Roman province until 27 BC. The Roman Senate granted Caesar a thanksgiving of 20 days for his victory in the Gallic War.[11]
"""


#MODEL 1
summarizerModel1 = pipeline("summarization", model="facebook/bart-large-cnn")

#print(summarizerModel1(TEXT, max_length=130, min_length=30, do_sample=False))


#MODEL 2
summarizerModel2 = pipeline("summarization", model="Falconsai/text_summarization")

#print(summarizerModel2(TEXT, max_length=130, min_length=30, do_sample=False))

#MODEL 3

summarizerModel3 = pipeline("summarization", model="pszemraj/led-large-book-summary")

#print(summarizerModel3(TEXT, max_length=130, min_length=30, do_sample=False))

#MODEL 4

summarizerModel4 = pipeline("summarization", model='it5/it5-base-news-summarization')

#print(summarizerModel4(TEXT, max_length=130, min_length=30, do_sample=False))


#Porównaj modele używając ROUGE-1 czyli ile się pokrywa słów

# Initialize the ROUGE-1 scorer
scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)


# Load the CNN/Daily Mail dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")

# Extract the first 20 documents and their reference summaries
for i, example in enumerate(dataset['train']):
    if i >= 5:
        break
    document = example['article']
    reference_summary = example['highlights']

    
    # Skracanie dokumentu (jeśli jest za długi modele go nie sparsują)
    max_doc_length = 1024
    if len(document.split()) > max_doc_length:
        document = ' '.join(document.split()[:max_doc_length])


    # Generate summaries using each model
    summary1 = summarizerModel1(document, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
    summary2 = summarizerModel2(document, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
    summary3 = summarizerModel3(document, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
    summary4 = summarizerModel4(document, max_length=130, min_length=30, do_sample=False)[0]['summary_text']


    # Print the document, reference summary, and generated summaries
    print(f"Document {i+1}:\n{document}\n")
    print(f"Reference Summary {i+1}:\n{reference_summary}\n")
    print(f"Model 1 Summary:\n{summary1}\n")
    print(f"Model 2 Summary:\n{summary2}\n")
    print(f"Model 3 Summary:\n{summary3}\n")
    print(f"Model 4 Summary:\n{summary4}\n")
    print("-" * 50)


    # Calculate ROUGE-1 scores
    scores1 = scorer.score(reference_summary, summary1)
    scores2 = scorer.score(reference_summary, summary2)
    scores3 = scorer.score(reference_summary, summary3)
    scores4 = scorer.score(reference_summary, summary4)

    # Print the ROUGE-1 scores
    print(f"Document {i+1} ROUGE-1 Scores:")
    print("Model 1 - Precision:", scores1['rouge1'].precision, "Recall:", scores1['rouge1'].recall, "F1 Score:", scores1['rouge1'].fmeasure)
    print("Model 2 - Precision:", scores2['rouge1'].precision, "Recall:", scores2['rouge1'].recall, "F1 Score:", scores2['rouge1'].fmeasure)
    print("Model 3 - Precision:", scores3['rouge1'].precision, "Recall:", scores3['rouge1'].recall, "F1 Score:", scores3['rouge1'].fmeasure)
    print("Model 4 - Precision:", scores4['rouge1'].precision, "Recall:", scores4['rouge1'].recall, "F1 Score:", scores4['rouge1'].fmeasure)
    print("-" * 50)


